# anchor生成示意图

原图片img首先经过卷积，得到fmap，

这个过程，如果只看尺寸，不看通道等，就仿佛是对img做了下采样

生成anchor的过程 是由一个小的卷积划窗，对fmap的每一个像素，都生成a个预设的窗口

如果fmap的尺寸为[b,c,w，h]那么，生成的窗口数目将与b，c无关，为$wha$个，a通常取值 m*n 或 m+n-1，m和n分别是锚框大小和宽高比类数。

设锚框大小为 $s \in (0,1]$，且宽高比为$r>0$，锚框的宽和高分别为 $ws \sqrt{r}$ 和$hs\sqrt{r}$。当中心位置给定时，已知宽和高的锚框是确定的。

anchor生成后，需要映射回原图上

# code

```python
from mxnet import contrib, image, nd
from matplotlib import  pyplot as plt
import gluonbook as gb

img = image.imread(os.path.join(gbpath,'img/catdog.jpg'))#.asnumpy()
h,w = img.shape[0:2]

def display_anchors(fmap_w, fmap_h, s):
    fmap = nd.zeros((1, 10, fmap_w, fmap_h))  # 前两维的取值不影响输出结果
    anchors = contrib.nd.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = nd.array((w, h, w, h))
    gb.show_bboxes(gb.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)

display_anchors(4,4,0.15)
display_anchors(2,2,0.4)

```
![](../img/anchor_demo.png)
![](../img/anchor_demo1.png)

# reference

参考mxnet教程
http://zh.d2l.ai/chapter_computer-vision/multiscale-object-detection.html