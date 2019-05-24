# Object detection

## Deep Learning for Generic Object Detection: A Survey

这是一篇arxiv长文 梳理了深度学习崛起后的DET任务 
[pdf](./DET_survey_liu2018deep/DET_survey_liu2018deep.pdf)

笔记：
 [【梗概】](./DET_survey_liu2018deep/DET_survey_liu2018deep(一).md) 
[【定义、挑战、历史】](./DET_survey_liu2018deep/DET_survey_liu2018deep(二).md)
[【算法框架：两阶段和一阶段】](./DET_survey_liu2018deep/DET_survey_liu2018deep(三).md)

## Recent Advances in Object Detection in the Age

[pdf](./age_advances/Recent%20Advances%20in%20Object%20Detection%20in%20the%20Age.pdf
)

这本书真的是一万个引用，我都不知道要放在哪里了 目前还是用到哪里看哪里的状态
主题就是目标检测，但是讲的非常全

# 目标检测算法

按照时间轴参考：

![](./img/deep_learning_object_detection_history.PNG)

https://github.com/hoya012/deep_learning_object_detection


这里经典的目标检测算法介绍可以参考 mxnet的某版本gluon教程[pdf:[动手学深度学习]9.7节](../book/)

也可以参考[DET_survey_liu2018deep.pdf](./DET_survey_liu2018deep/DET_survey_liu2018deep.pdf)的[第三章](./DET_survey_liu2018deep/DET_survey_liu2018deep(三).md)

以及[Recent Advances in Object Detection in the Age](./age_advances)的2.1章节

下面是我自己的整理，主要参考的gluon和原文

## 两阶段

[[两阶段目标检测方法综述]](./2/)


## 一阶段

[jump to...](./1)

## 一些概念解释

* RPN
  
* anchor

[[RPN原理与anchor]](./feynman/RPN_anchor.md)

[[有anchor目标检测中anchor生成示意]](./feynman/anchor_demo.md)

* dataset
  
[[目标检测数据集]](./feynman/datasets.md)

* IoU
  
[[评价指标简介]](./feynman/IoU.md)

* mAP

[[目标检测数据集的指标和实现（todo）]](./feynman/mAP.md)

* NMS
  
[[NMS:非极大值抑制]](./feynman/NMS.md)

* SPP

* ROI

[[RoI Pooling:感兴趣区域池化]](./feynman/ROI.md)



[jump to...](./feynman)