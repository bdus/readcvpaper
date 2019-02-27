# Evaluation measures

评价指标是量化衡量算法表现的一个手段，表现为一系列值。

指标由在线的（Online metrics）和离线的（Offline metrics）

## 错误率和精度
分类任务中常用的两种性能度量：

错误率：
$$
E(f;D) = \frac{1}{m} \sum_{i=1}^{m} \mathbb I(f(x_i) \not=y_i)
$$
![](https://latex.codecogs.com/png.latex?E%28f%3BD%29%20%3D%20%5Cfrac%7B1%7D%7Bm%7D%20%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Cmathbb%20I%28f%28x_i%29%20%5Cnot%3Dy_i%29)

精度定义为

$$
\begin{aligned}
acc(f;D) &=  \frac{1}{m}\sum_{i=1}^{m} \mathbb I (f(x_i)=y_i)  \\
 &= 1-E(f;D)
\end{aligned}
$$

![](https://latex.codecogs.com/png.latex?%5Cbegin%7Balign*%7D%20acc%28f%3BD%29%20%26%3D%20%5Cfrac%7B1%7D%7Bm%7D%5Csum_%7Bi%3D1%7D%5E%7Bm%7D%20%5Cmathbb%20I%20%28f%28x_i%29%3Dy_i%29%20%5C%5C%20%26%3D%201-E%28f%3BD%29%20%5Cend%7Balign*%7D)


## 查准率、查全率与F1

查准率（precision） 查全率（recall）
查准率 衡量的类似于 "检索出的结果有多少比率是用户感兴趣/相关的"

![](https://wikimedia.org/api/rest_v1/media/math/render/svg/95eeb143dd5ded21c882eaa4530ec69739a3a218)

可以看西瓜书P30的例子
二分的错误有两种
FN 假反例 真实情况为正例的分到反例
FP 假正例 真是为反例的分到正例

**查准率**：

$$
P=\frac{TP}{TP+FP}
$$
也就是 真正例 占 预测结果为正例的比例，检索出来的信息（结果为正例）中有多少是用户感兴趣的（真正例）


**查全率**

$$
R = \frac{TP}{TP+FN}
$$
也就是 真正例 占 标签正例的比例，用户感兴趣的信息（真实正例）中有多少被检索出来（真正例）了

对于分类结果有一个$n \times n$的混淆矩阵 用于衡量



![](https://img-blog.csdn.net/20150407223936418?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdmVzcGVyMzA1/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

### P-R曲线

P-R曲线 就是 精确率precision vs 召回率recall 曲线
P和R都可以利用混淆矩阵计算出来
那如何绘制PR曲线呢？
![](https://img-blog.csdn.net/20180716161513928?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzI3ODcxOTcz/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
西瓜书上说
>我们可根据学习器的预测结果对样例进行排序，排在前面的是学习器认为最可能是正例的样本，排在后面的……。**按此顺序逐个把样本作为正例进行预测**，则每次可以计算出当前的查全率、查准率，以此描点绘图。

这段话当时没看懂，关键就是黑体部分

>算法对样本进行分类时，都会有置信度，即表示该样本是正样本的概率，比如99%的概率认为样本Ａ是正例，１％的概率认为样本B是正例。通过选择合适的阈值，比如50%，对样本进行划分，概率大于50%的就认为是正例，小于50%的就是负例。

所以黑体部分就是条件正例划分阈值的一个过程，所以P-R曲线的平衡点可以作为阈值选取的依据。

**平衡点 Break-Even Point**
就是查准率=查全率的时候的取值

### F1
F1是基于查准率和查全率的调和平均定义的度量 西瓜书P32

$$
F1=\frac{2 \times P \times R}{P+R}
$$

## ROC与AUC

ROC曲线和PR曲线绘制方式类似，区别的横纵坐标值分别是 
“真正例率” True Positive Rate 和 ”假正例率“ False Positive Rate

$$
TPR=\frac{TP}{TP+FN}
$$

$$
FPR=\frac{FP}{TN+FP}
$$


AUC: Area Under ROC Curve
是ROC曲线下的面积，用于衡量学习器性能

# 检测中的Metric

## AP、mAP
平均精度 AP： Average precision
AP 就是PR曲线下方的面积,一般越好的分类器AP值越高
![](https://wikimedia.org/api/rest_v1/media/math/render/svg/1cff0cc58e4629105e6066df9e07d567a1784d44)

mAP  mean Average Precision

mAP是多个类别AP的平均值。这个mean的意思是对每个类的AP再求平均，得到的就是mAP的值，mAP的大小一定在[0,1]区间，越大越好。该指标是目标检测算法中最重要的一个。

## IoU

但是目标检测任务中，不仅预测了目标的种类，还预测了目标的位置,在计算PR的时候和分类是不一样的
目标检测任务需要预测出目标的位置 并且给出类别

目标检测的Ground Truth 一张图片包括多个目标 一个目标有5个参数：

    类别 、X、Y、Box Width、Box Height

检测框与真实框重叠区域就是交集区域 预测框与真实框的总面积区域就是并集框

![](https://img-blog.csdn.net/20180218093846965?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS2F0aGVyaW5lX2hzcg==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

当目标位置检测准确的时候，检测框与GT框重合，IoU应该也为1

同样地，在使用IoU值看检测是否正确的时候,也要选取一个阈值做比较
常用的阈值为0.5
目标检测中的

    TP：IoU>0.5的检测框数量（同一个GT只计算一次）
    FP：IoU<=0.5的检测框，或者是检测到同一个GT的多余检测框的数量
    FN：没有检测到的GT的数量

同样地

$$
P=\frac{TP}{TP+FP}
$$


$$
R = \frac{TP}{TP+FN}
$$

可以看出，TP+FP就等于检测框数量，TP+FN就是GT的数量

# reference

[[Wikipedia:Evaluation measures]](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
西瓜书