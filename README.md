# [Automatic-Quantification-of-Plant-Disease-From-Field-Image-Data-Using-Deep-Learning](https://openaccess.thecvf.com/content/WACV2021/papers/Garg_Automatic_Quantification_of_Plant_Disease_From_Field_Image_Data_Using_WACV_2021_paper.pdf)

## Our Contributions
* The generation of disease region and leaf instance
masks in the existing dataset.
* An end-to-end deep learning framework with multitask loss function for simultaneous segmentation of
leaf and diseased region using unified feature maps.
* To the best of our knowledge, this is the first study to
quantify disease severity corresponding to individual
leaves from UAV field images.



## Data
[Data](https://drive.google.com/file/d/1uIb7S9R8H-2gsD-491CK0qDfUIMgO_FJ/view?usp=sharing) is labelled under the supervision of the experts. Kindly do cite if you use our data or find our results/code useful in your research:

```
@InProceedings{Garg_2021_WACV,
    author    = {Garg, Kanish and Bhugra, Swati and Lall, Brejesh},
    title     = {Automatic Quantification of Plant Disease From Field Image Data Using Deep Learning},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1965-1972}
}
```

## Cascade MRCNN
An End to End Pipeline for Cascaded Instance Segmentation built on the top of MaskRCNN code.

![alt text](https://github.com/kanishgarg/Cascade-MRCNN/blob/master/Cascade%20MRCNN%20Architecture.png)

Figure 1. Cascaded MRCNN framework. Given an input image (i) Primary mask branch predicts individual leaf segments (Leaf instance
segmentation) and a (ii) Cascaded mask branch is added that utilises the feature maps corresponding to leaf instances for generating
diseased region mask


Related Work: [```A Hierarchical Framework for Leaf Instance Segmentation: Application to Plant Phenotyping```](https://ieeexplore.ieee.org/abstract/document/9411981)


