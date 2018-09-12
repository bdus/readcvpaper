package tritrain;

/**
 * Description: Use tri-training to exploit unlabeled data.
 *
 * Reference:   Z.-H. Zhou, Y. Jiang, M. Li. Tri-Training: exploiting unlabeled data using three classifiers.
 *              IEEE Transactions on Knowledge and Data Engineering. in press.
 *
 * ATTN:        This package is free for academic usage. You can run it at your own risk.
 *	     	For other purposes, please contact Prof. Zhi-Hua Zhou (zhouzh@nju.edu.cn).
 *
 * Requirement: To use this package, the whole WEKA environment (ver 3.4) must be available.
 *	        refer: I.H. Witten and E. Frank. Data Mining: Practical Machine Learning
 *		Tools and Techniques with Java Implementations. Morgan Kaufmann,
 *		San Francisco, CA, 2000.
 *
 * Data format: Both the input and output formats are the same as those used by WEKA.
 *
 * ATTN2:       This package was developed by Mr. Ming Li (lim@lamda.nju.edu.cn). There
 *		is a ReadMe file provided for roughly explaining the codes. But for any
 *		problem concerning the code, please feel free to contact with Mr. Li.
 *
 */

import java.io.*;
import java.util.*;

import weka.core.*;
import weka.classifiers.Classifier;
import weka.classifiers.trees.*;


public class TriTrain
{
  /** base classifiers */
  private Classifier m_baseClassifier = null;

  /** classifiers */
  private Classifier[] m_classifiers;

  private Random m_rand = new Random(1);


  /**
   * The constructor
   */
  public TriTrain()
  {
  }


  /**
   * Sets the base classifier
   *
   * @param c Classifier The base classifier
   */
  public void setClassifier(Classifier c)
  {
    m_baseClassifier = c;
  }

  public void setRandomObject(Random random)
  {
    m_rand = random;
  }

  /**
   * Builds classifier via tri-training
   *
   * @param labeled Instances The labeled set
   * @param unlabeled Instances The unlabeled set
   * @throws Exception Some exception
   */
  public void buildClassifier(Instances labeled, Instances unlabeled) throws Exception
  {
    double[] err = new double[3];             // e_i
    double[] err_prime = new double[3];       // e'_i
    double[] s_prime = new double[3];         // l'_i

    if (m_baseClassifier == null)
      throw new Exception("Base classifier should be set before the building process");

    if (!labeled.classAttribute().isNominal())
      throw new Exception("The class value should be nominal");

    m_classifiers = Classifier.makeCopies(m_baseClassifier, 3);
    Instances[] labeleds = new Instances[3];

    for(int i = 0; i < 3; i++)
    {
      labeleds[i] = new Instances(labeled.resampleWithWeights(m_rand));     //L_i <-- Bootstrap(L)
      m_classifiers[i].buildClassifier(labeleds[i]);                        //h_i <-- Learn(L_i)
      err_prime[i] = 0.5;                                                   //e'_i <-- .5
      s_prime[i] = 0;                                                       //l'_i <-- 0
    }

    boolean bChanged = true;

    /** repeat until none of h_i ( i \in {1...3} ) changes */
    while(bChanged)
    {
      bChanged = false;
      boolean[] bUpdate = new boolean[m_classifiers.length];
      Instances[] Li = new Instances[3];

      /** for i \in {1...3} do */
      for(int i = 0; i < 3; i++)
      {
        Li[i] = new Instances(labeled, 0);         //L_i <-- \phi
        err[i] = measureError(labeled, i);         //e_i <-- MeasureError(h_j & h_k) (j, k \ne i)

        /** if (e_i < e'_i) */
        if(err[i] < err_prime[i])
        {
          /** for every x \in U do */
          for(int j = 0; j < unlabeled.numInstances(); j++)
          {
            Instance curInst = new Instance(unlabeled.instance(j));
            curInst.setDataset(Li[i]);
            double classval = m_classifiers[(i+1)%3].classifyInstance(curInst);

            /** if h_j(x) = h_k(x) (j,k \ne i) */
            if(classval == m_classifiers[(i+2)%3].classifyInstance(curInst))
            {
              curInst.setClassValue(classval);
              Li[i].add(curInst);                //L_i <-- L_i \cup {(x, h_j(x))}
            }
          }// end of for j

          /** if (l'_i == 0 ) */
          if(s_prime[i] == 0)
            s_prime[i] = Math.floor(err[i] / (err_prime[i] - err[i]) + 1);   //l'_i <-- floor(e_i/(e'_i-e_i) +1)

          /** if (l'_i < |L_i| ) */
          if(s_prime[i] < Li[i].numInstances())
          {
            /** if ( e_i * |L_i| < e'_i * l'_i) */
            if(err[i] * Li[i].numInstances() < err_prime[i] * s_prime[i])
              bUpdate[i] = true;                                          // update_i <-- TURE

            /** else if (l'_i > (e_i / (e'_i - e_i))) */
            else if (s_prime[i] > (err[i] / (err_prime[i] - err[i])))
            {
              int numInstAfterSubsample = (int) Math.ceil(err_prime[i] * s_prime[i] / err[i] - 1);
              Li[i].randomize(m_rand);
              Li[i] = new Instances(Li[i], 0, numInstAfterSubsample);         //L_i <-- Subsample(L_i, ceilling(e'_i*l'_i/e_i-1)
              bUpdate[i] = true;                                              //update_i <-- TRUE
            }
          }
        }
      }//end for i = 1...3

      //update
      for(int i = 0; i < 3; i++)
      {
        /** if update_i = TRUE */
        if(bUpdate[i])
        {
          int size = Li[i].numInstances();
          bChanged = true;
          m_classifiers[i].buildClassifier(combine(labeled, Li[i]));        //h_i <-- Learn(L \cup L_i)
          err_prime[i] = err[i];                                            //e'_i <-- e_i
          s_prime[i] = size;                                                //l'_i <-- |L_i|
        }
      }// end fo for
    } //end of repeat
  }

  /**
   * Returns the probability label of a given instance
   *
   * @param inst Instance The instance
   * @return double[] The probability label
   * @throws Exception Some exception
   */
  public double[] distributionForInstance(Instance inst) throws Exception
  {
    double[] res = new double[inst.numClasses()];
    for(int i = 0; i < m_classifiers.length; i++)
    {
      double[] distr = m_classifiers[i].distributionForInstance(inst);
      for(int j = 0; j < res.length; j++)
        res[j] += distr[j];
    }
    Utils.normalize(res);
    return res;
  }

  /**
   * Classifies a given instance
   *
   * @param inst Instance The instance
   * @return double The class value
   * @throws Exception Some Exception
   */
  public double classifyInstance(Instance inst) throws Exception
  {
    double[] distr = distributionForInstance(inst);
    return Utils.maxIndex(distr);
  }

  /**
   * Adds the instances in initial training set L to the newly labeled set Li
   *
   * @param L Instances The initial training set
   * @param Li Instances The newly labeled set
   * @return Instances The combined data set
   */
  private Instances combine(Instances L, Instances Li)
  {
    for(int i = 0; i < L.numInstances(); i++)
      Li.add(L.instance(i));

    return Li;
  }

  /**
   * Measure combined error excluded the classifier 'id' on the given data set
   *
   * @param data Instances The data set
   * @param id int The id of classifier to be excluded
   * @return double The error
   * @throws Exception Some Exception
   */
  protected double measureError(Instances data, int id) throws Exception
  {
    Classifier c1 = m_classifiers[(id+1)%3];
    Classifier c2 = m_classifiers[(id+2)%3];
    double err = 0;
    int count = 0;

    for(int i = 0; i < data.numInstances(); i++)
    {
      double prediction1 = c1.classifyInstance(data.instance(i));
      double prediction2 = c2.classifyInstance(data.instance(i));

      if(prediction1 == prediction2)
      {
        count++;
        if(prediction1 != data.instance(i).classValue())
          err += 1;
      }
    }

    err /= count;
    return err;
  }


  /**
   * The main method just for demonstration how to use this class
   *
   * @param args String[]
   */
  public static void main(String[] args)
  {
    try
    {
      BufferedReader r = new BufferedReader(new FileReader("labeled.arff"));
      Instances labeled = new Instances(r);
      labeled.setClassIndex(labeled.numAttributes()-1);
      r.close();

      r = new BufferedReader(new FileReader("unlabeled.arff"));
      Instances unlabeled = new Instances(r);
      unlabeled.setClassIndex(unlabeled.numAttributes()-1);
      r.close();

      r = new BufferedReader(new FileReader("test.arff"));
      Instances test = new Instances(r);
      test.setClassIndex(test.numAttributes()-1);
      r.close();

      J48 dt = new J48();
      dt.setUnpruned(true);

      TriTrain tri = new TriTrain();
      tri.setClassifier(dt);
      tri.buildClassifier(labeled, unlabeled);

      double err = 0;
      for(int i = 0; i < test.numInstances(); i++)
      {
        if (tri.classifyInstance(test.instance(i)) != test.instance(i).classValue())
          err += 1.0;
      }
      err /= test.numInstances();

      System.out.println("Error rate = " + err);
    }
    catch(Exception e)
    {
      e.printStackTrace();
    }
  }

}
