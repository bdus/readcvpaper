# RPN&anchor

RPN(Region Proposal network)和anchor是Faster RCNN中重要的概念,在其它检测算法中也被频繁地使用,这里简单说一下原理,主要实现一下.

我们曾经在[link](https://github.com/bdus/readcvpaper/blob/master/DET/2/readme.md#rpn)讲Faster-RCNN时候提到了一点,回顾一下:

![](../img/rpn.JPG)

1. 卷积网络抽取的特征首先进入一个填充数为 1、通道数为 256 的 3 × 3 卷积层,这样每个像
素得到一个 256 ⻓度的特征表示。
2. 以每个像素为中心,**生成多个大小和比例不同的锚框和对应的标注**。每个锚框使用其中心
像素对应的 256 维特征来表示。
3. 在锚框特征和标注上面训练一个两类分类器,判断其含有感兴趣目标还是只有背景。
4. 对每个被判断成含有目标的锚框,进一步预测其边界框,然后进入 RoI 池化层。

对于每个anchor,根据其尺寸和特征向量分支出两个全连接层: cls-layer 和 reg-layer
1. cls-layer 输出是bool,用于判断这个Proposal有没有object(是前景还是背景)
2. reg-layer 输出有四个,用于预测proposal的中心锚点对应的(x,y,w,h)

不管懂没懂,先来写一下没有错的.

因此RPN的代码可以拆分一下:
    首先是以每个像素为中心,**生成多个大小和比例不同的锚框和对应的标注** 
    然后是对每个像素提256d特征, 连接cls-layer

// 填坑 # 先停工