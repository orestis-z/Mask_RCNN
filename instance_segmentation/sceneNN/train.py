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

# Path to ADE20K trained weights
ADE20K_MODEL_PATH = os.path.join(ROOT_DIR, "logs/seg_ade20k20171109T1726/mask_rcnn_seg_ade20k_0018.h5")
WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/seg_scenenn20171109T1726/mask_rcnn_seg_scenenn_0153.h5")

SCENENN_DIR = "/external_datasets/sceneNN"

config = Config()
config.display()

# Create model in training mode
print('creating model..')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

exclude = ["conv1"]

# # Which weights to start with?
init_with = "last"  # ade20k or last

print('loading weights...')
if init_with == "ade20k":
    model.load_weights(ADE20K_MODEL_PATH, by_name=True,
                   exclude=exclude)
if init_with == "custom":
    model.load_weights(WEIGHTS_PATH, by_name=True)
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

# Training dataset
dataset_train = Dataset()
dataset_train.load(SCENENN_DIR, "training")
dataset_train.prepare()

# Validation dataset
dataset_val = Dataset()
dataset_val.load(SCENENN_DIR, "validation")
dataset_val.prepare()

# Fine tune all layers
print('fine tuning all layers...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=1000,
            layers='all')
            # layers='|'.join(exclude))
