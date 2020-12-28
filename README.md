# Cascade-MRCNN
An End to End Pipeline for Cascaded Instance Segmentation built on the top of MaskRCNN code.

![alt text](https://github.com/kanishgarg/Cascade-MRCNN/blob/master/Cascade%20MRCNN%20Architecture.png)

Figure 1. Cascaded MRCNN framework. Given an input image (i) Primary mask branch predicts individual leaf segments (Leaf instance
segmentation) and a (ii) Cascaded mask branch is added that utilises the feature maps corresponding to leaf instances for generating
diseased region mask
