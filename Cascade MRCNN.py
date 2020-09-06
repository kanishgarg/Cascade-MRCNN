
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib
import mrcnn.visualize as visualize
from mrcnn.model import log
import mcoco.coco as coco
import mextra.utils as extra_utils
import keras
import warnings
warnings.filterwarnings('ignore')

HOME_DIR = '.'
DATA_DIR = os.path.join(HOME_DIR, "pycococreator/train")
MODEL_DIR = os.path.join(DATA_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(HOME_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# # Dataset
# 
# Organize the dataset using the following structure:
# 
# ```
# DATA_DIR
# │
# └───annotations
# │   │   instances_<subset><year>.json
# │   
# └───<subset><year>
#     │   image021.jpeg
#     │   image022.jpeg
# ```

# In[4]:


dataset_train = coco.CocoDataset()
dataset_train.load_coco(DATA_DIR, subset="plants_train", year="2019")
dataset_train.prepare()

dataset_validate = coco.CocoDataset()
dataset_validate.load_coco(DATA_DIR, subset="plants_validate", year="2019")
dataset_validate.prepare()

# dataset_test = coco.CocoDataset()
# dataset_test.load_coco(DATA_DIR, subset="shapes_test", year="2018")
# dataset_test.prepare()


image_size = 384
rpn_anchor_template = (1, 2, 4, 8, 16) # anchor sizes in pixels
rpn_anchor_scales = tuple(i * (image_size // 16) for i in rpn_anchor_template)

class ShapesConfig(Config):
    """Configuration for training on the shapes dataset.
    """
    NAME = "shapes"

    # Train on 1 GPU and 2 images per GPU. Put multiple images on each
    # GPU if the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 16

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes (triangles, circles, and squares)

    # Use smaller images for faster training. 
    IMAGE_MAX_DIM = image_size
    IMAGE_MIN_DIM = image_size
    
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = rpn_anchor_scales

    # Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

#     STEPS_PER_EPOCH = 400
    STEPS_PER_EPOCH = 400

    VALIDATION_STEPS = 10
    
config = ShapesConfig()
config.display()


# # Model
# 

model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)


inititalize_weights_with = "coco"  # imagenet, coco, or last

if inititalize_weights_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
    
elif inititalize_weights_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
     
elif inititalize_weights_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


import imgaug
augmentation = imgaug.augmenters.Sometimes(0.8, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.Flipud(0.5),
                    # imgaug.augmenters.Rot90(0.5),
                    # imgaug.augmenters.Affine(0.5, scale=(0.5, 1.5)),
                    # imgaug.augmenters.PerspectiveTransform(0.5,scale=(0.01, 0.15)),
                    # imgaug.augmenters.CropAndPad(0.5,percent=(-0.25, 0.25)),

                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
# 
# # Training
# 
# Training in two stages

# ## Heads
# 
# Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass layers='heads' to the train() function.
# 
# ## Fine-tuning
# 
# Fine-tune all layers. Pass layers="all to train all layers.



# 1000, 200
model.train(dataset_train, dataset_validate, 
            learning_rate=config.LEARNING_RATE, 
            epochs=50,
            layers='heads', augmentation=augmentation)

model.train(dataset_train, dataset_validate, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=70, # starts from the previous epoch, so only 1 additional is trained 
            layers="all",augmentation=augmentation)
