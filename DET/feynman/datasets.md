# 目标检测数据集

总结一下目标检测相关数据集

这些数据集都可以在
    https://gluon-cv.mxnet.io/build/examples_datasets/index.html
获取


## 经典通用目标数据集

### Pascal VOC

[【VOC】](http://host.robots.ox.ac.uk/pascal/VOC/)

Pascal VOC 是最经典的数据集之一，也是现在许多目标检测算法常用的测试集，据说新算法已经开始在上面过拟合了。

VOC的0.5IoU TP判别标准的测试指标 现在也被各种目标检测算法使用着。

类型:
Person: person; 
Animal: bird,cat, cow, dog, horse, sheep;
Vehicle: aeroplane, bicycle, boat, bus, car, motor-bike, train;
Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor

通常说的VOC数据集包括两个版本，VOC2007 与 VOC2012

- 所有的标注图片都有Detection需要的label， 但只有部分数据有Segmentation Label。 
- VOC2007中包含9963张标注过的图片， 由train/val/test三部分组成， 共标注出24,640个物体。 
- VOC2007的test数据label已经公布， 之后的没有公布（只有图片，没有label） 
- 对于检测任务，VOC2012的trainval/test包含08-11年的所有对应图片。 trainval有11540张图片共27450个物体。

数据集下载完后会有5个文件夹。Annotations、ImageSets、JPEGImages、SegmentationClass、SegmentationObject。



### MS COCO

是现在比较有挑战的一个数据集。

[【Leader board】](http://cocodataset.org/#detection-leaderboard)

118000 training images， 5000 validation images， 41000 testing images
此外还有120K的未标注图片，和有标签图像服从同样分布,可用于半监督训练

其数据集共80类，是VOC的四倍

AP计算和VOC类似，但是 IoU从0.5变为0.95 具体不一样 看[link](http://cocodataset.org/#detection-eval)


### ImageNet DET

ImageNet DET的词汇是通过WordNet的名词组织的每一层次的节点都有平均5000张照片描述
LSVRC是基于ImageNet组织的挑战，检测任务目标种类总共有200类，除了分类、还有DET，localization，VID任务

# reference

Recent Advances in Object Detection in the Age
VOC https://blog.csdn.net/liuxiao214/article/details/80552026 

