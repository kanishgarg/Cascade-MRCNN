
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


# In[5]:


# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 4)
# print(image_ids)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, mask2, class_ids = dataset_train.load_mask(image_id)
#     print(mask.shape)
#     print(mask2.shape)
#     print(np.unique(mask2[:,:,0], return_counts=True))
#     visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)


# # Configuration

# In[5]:


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

# In[7]:


model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)


# In[8]:


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
# model.load_weights(MODEL_PATH2, by_name=True)


# In[9]:


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
# augmentation=None


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

# In[ ]:


# 1000, 200
model.train(dataset_train, dataset_validate, 
            learning_rate=config.LEARNING_RATE, 
            epochs=100,
            layers='heads', augmentation=augmentation)

model.train(dataset_train, dataset_validate, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=110, # starts from the previous epoch, so only 1 additional is trained 
            layers="all",augmentation=augmentation)


# # Detection

# In[8]:


# class InferenceConfig(ShapesConfig):
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# inference_config = InferenceConfig()

# # Recreate the model in inference mode
# model = modellib.MaskRCNN(mode="inference", 
#                           config=inference_config,
#                           model_dir=MODEL_DIR)

# # Get path to saved weights
# # Either set a specific path or find last trained weights
# # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# print(model.find_last())


# # In[10]:


# model_path = model.find_last()

# # Load trained weights (fill in path to trained weights here)
# assert model_path != "", "Provide path to trained weights"
# print("Loading weights from ", model_path)
# model.load_weights(model_path, by_name=True)


# # ### Test on a random image from the test set
# # 
# # First, show the ground truth of the image, then show detection results.

# # In[21]:


# #for i in range(len(dataset_validate.image_ids)):

# image_id = random.choice(dataset_validate.image_ids)
# #image_id = i
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask, gt_mask2 =    modellib.load_image_gt(dataset_validate, inference_config, 
#                            image_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
# log("gt_mask2", gt_mask2)
# plt.imshow(original_image)
# cv2.imwrite("results/img" +str(image_id) +".jpg", original_image)
# plt.show()
# res_type = "gth"
# # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_mask2, gt_class_id, 
# #                             dataset_validate.class_names, figsize=(8, 8),show_mask=False, show_mask2=True, show_bbox=False)

# results = model.detect([original_image], verbose=1)

# r = results[0]
# # print(r['masks'][0])
# visualize.display_instances(original_image, r['rois'], r['masks'], r['masks2'], r['class_ids'], 
#                             dataset_validate.class_names, r['scores'], ax=get_ax(),show_mask=True, show_mask2=False, show_bbox=False)
# # res_type = "pred"
# # visualize.save_image(original_image, r['rois'], r['masks'], r['class_ids'], 
# #                            dataset_validate.class_names, r['scores'], ax=get_ax(),ipath= "results/img",image_id= image_id,res_type= res_type)


# # # Evaluation
# # 
# # Use the test dataset to evaluate the precision of the model on each class. 

# # In[43]:


# predictions =extra_utils.compute_multiple_per_class_precision(model, inference_config, dataset_train,
#                                                  number_of_images=500, iou_threshold=0.5)
# complete_predictions = []

# for shape in predictions:
#     complete_predictions += predictions[shape]
#     print("{} ({}): {}".format(shape, len(predictions[shape]), np.mean(predictions[shape])))

# print("--------")
# print("average: {}".format(np.mean(complete_predictions)))


# # ## Convert result to COCO
# # 
# # Converting the result back to a COCO-style format for further processing 

# # In[87]:


# import json
# import pylab
# import matplotlib.pyplot as plt
# from tempfile import NamedTemporaryFile
# from pycocotools.coco import COCO

# coco_dict = extra_utils.result_to_coco(results[0], dataset_validate.class_names,
#                                        np.shape(original_image)[0:2], tolerance=0)

# with NamedTemporaryFile('w') as jsonfile:
#     json.dump(coco_dict, jsonfile)
#     jsonfile.flush()
#     coco_data = COCO(jsonfile.name)


# # In[93]:


# category_ids = coco_data.getCatIds(catNms=['square', 'circle', 'triangle'])
# image_data = coco_data.loadImgs(1)[0]
# image = original_image
# plt.imshow(image); plt.axis('off')
# pylab.rcParams['figure.figsize'] = (8.0, 10.0)
# annotation_ids = coco_data.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
# annotations = coco_data.loadAnns(annotation_ids)
# coco_data.showAnns(annotations)


# # In[68]:


# image_id = random.choice(dataset_validate.image_ids)
# original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_validate, inference_config, 
#                            image_id, use_mini_mask=False)

# log("original_image", original_image)
# log("image_meta", image_meta)
# log("gt_class_id", gt_class_id)
# log("gt_bbox", gt_bbox)
# log("gt_mask", gt_mask)
# plt.imshow(original_image)
# plt.show()
# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
#                             dataset_validate.class_names, figsize=(8, 8))

# results = model.detect([original_image], verbose=1)

# r = results[0]
# # print(r['masks'][0])
# visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
#                             dataset_validate.class_names, r['scores'], ax=get_ax())


# # In[ ]:





# # In[ ]:




