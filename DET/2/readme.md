# 两阶段方法

我们这里只梳理算法的主要工作，不说细节。

## R-CNN

R-CNN的框架对后续算法影响十分的大,其主要贡献在于采用了预训练好的卷积神经网络来对抽取特征，取得了比较好的精度。

[](../img/rcnn.svg)

1. 首先利用Selective Research选择出图片上可能有目标的候选区域。
2. 选择一个预先训练好的卷积神经网络，去掉最后的输出层作为提特征提取模块。将候选区域resize成卷积网络需要的输入尺寸，然后提取特征。
3. 将每个提议区域的特征连同其标注做成一个样本,训练多个支持向量机(SVM)来进行目标类别分类,这里第 i 个 SVM 预测样本是否属于第 i 类。(注意这里是n个独立的SVM分类器)
4. 在这些样本上训练一个线性回归模型来预测真实边界框。

[pdf: rcnn](./DET1_girshick2014rich.pdf)


## SPPNet 和 Fast R-CNN

R-CNN的主要性能瓶颈在于每个候选区域，都要做一次网络的前向计算,Fast R-CNN 的主要改进是根据图片的feature map保持空间位置的相对一致性，引入ROI的映射和pooling。

[](../img/fast-rcnn.svg)

主要改进：
1. 用来提取特征的卷积网络是作用在整个图像上,而不是各个提议区域上。而且这个卷积网络通常会参与训练,即更新权重。
2. 选择性搜索是作用在卷积网络的输出上,而不是原始图像上。
3. 在 R-CNN 里,我们将形状各异的提议区域变形到同样的形状来进行特征提取。Fast R-CNN则新引入了兴趣区域池化层(Region of Interest Pooling,简称 RoI 池化层)来对每个提议区域提取同样大小的输出以便输入之后的神经层。
4. 在目标分类时,Fast R-CNN 不再使用多个 SVM,而是像之前图像分类那样使用 Softmax 回归来进行多类预测。

[pdf: fast-rcnn](./Girshick_Fast_R-CNN_ICCV_2015_paper.pdf)


实际上RoI pooling就是SPPNet只有一层的情况。
关键在于feature map的映射和ROI pooling

### 1. feature map的映射

为了说明这个，从dff盗一张图
![](../img/featuremap.JPG)

图中是关键/非关键帧的特征图的可视化，可以看到#183卷积核学习到的可能是车的特征，#289卷积核学习的可能是行人的特征，关键是，你会发现卷积特征的位置和原图的位置是相关的，可以说是相对应的。

这就是feature map 映射操作的基础，我们之前要从一张图中抽取候选框，然后再计算feature 现在 **原始图像的框，可以对应于卷积图像相同位置的框** 因此我们只需要对整张图提取一次卷积，然后根据原图的候选框选择，在卷积图上找到对应，就可以节省大量的计算。

### 2. ROI pooling

ROI的作用，就是不管你输入多少，我的输出数目都是固定的。

SPPNet就是对一张图做不同尺度的ROI 然后把特征顺序连接起来,这样不管你输入图片多大尺寸，我输出的特征都一样长

![](../img/SPPNet.JPG)

详细请参考论文：
[pdf: SPPNet](./SPPNet.pdf)
或我在别处的说明[[ROI]](../feynman/ROI.md)


## Faster-RCNN

Faster-RCNN提出了RPN网络，终于，Region Proposal也由网络来完成了

为此，引入了anchor、和RPN的概念

[](../img/faster-rcnn.svg)
