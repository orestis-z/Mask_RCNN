import os, sys
import tensorflow as tf
import re

from dataset import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import model as modellib
from model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
SCENENN_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_seg_scenenn_0101.h5")

SCENENET_DIR = "/external_datasets/SceneNet_RGBD"

config = ObjectsConfig()
config.display()

# Training dataset
dataset_train = ObjectsDataset()
dataset_train.load_sceneNet(SCENENET_DIR, "training")
dataset_train.prepare()

# Validation dataset
dataset_val = ObjectsDataset()
dataset_val.load_sceneNet(SCENENET_DIR, "validation")
dataset_val.prepare()

# Create model in training mode
print('creating model..')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

exclude = []

# # Which weights to start with?
init_with = "last"  # scenenn or last

print('loading weights...')
if init_with == "scenenn":
    model.load_weights(SCENENN_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.
# Train the head branches
# print('training heads...')
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE, 
#             epochs=j, 
#             layers='heads')

# Fine tune all layers
print('fine tuning all layers...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=1000,
            layers='all')
