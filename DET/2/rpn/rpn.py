#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:52:25 2018

@author: bdus

@Describe: implemention of RPN in Faster RCNN

"""

import numpy as np
import matplotlib.pyplot as plt
'''
parameters
'''
# size of feature map
size_feat_y = 16
size_feat_x = 16
# downsample (pixels at src img / per pixel at feat map)
rpn_stride = 8
# anchor scales&ratios
scales = [1,3,5]
ratios = [0.5,1,2]

"""
# get grid
f_x = np.arange(4)
f_y = np.arange(4)
f_x

F_X, F_Y = np.meshgrid(f_x,f_y)

# get size

scale, ratio = np.meshgrid(scales,ratios)
scale = scale.flatten()
ratio = ratio.flatten()
"""

def anchor_gen(size_feat_x,size_feat_y,rpn_stride,scales,ratios):
    """
    4x4 feature map have 4*4=16 anchor-points
    for each location there are 9 type of anchors
    there total 9x16 anchors 
    9x16 == centerX.shape == anchorX.shape
            
    """
    scales, ratios = np.meshgrid(scales,ratios)
    scales, ratios = scales.flatten(), ratios.flatten()
    #width and height of anchor
    scalesY = scales * np.sqrt(ratios) 
    scalesX = scales / np.sqrt(ratios)
    #point of anchor
    shiftX = np.arange(0,size_feat_x) * rpn_stride
    shiftY = np.arange(0,size_feat_y) * rpn_stride
    shiftX,shiftY = np.meshgrid(shiftX,shiftY)
    #get all combine   anchors
    centerX,anchorX = np.meshgrid(shiftX,scalesX)
    centerY,anchorY = np.meshgrid(shiftY,scalesY)
    #
    anchor_center = np.stack([centerY,centerX],axis=2).reshape(-1,2)
    anchor_size = np.stack([anchorY,anchorX],axis=2).reshape(-1,2)
    boxes = np.concatenate([anchor_center - 0.5*anchor_size, anchor_center+ 0.5*anchor_size],axis=1)
    return boxes

anchors = anchor_gen(size_feat_x,size_feat_y,rpn_stride,scales,ratios)
anchors.shape

import matplotlib.patches as patches

plt.figure(figsize=[10,10])
img = np.ones((128,128,3))
plt.imshow(img)

Axs = plt.gca() ## get current Axs

for i in range(anchors.shape[0]):
    box = anchors[i]
    rec = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],edgeColor="r",facecolor="none")
    #color_list = ['b','g','r','c','m','y','k','w']
    #rec = patches.Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],edgeColor=color_list[i%8] ,facecolor="none")
    Axs.add_patch(rec)

plt.show()