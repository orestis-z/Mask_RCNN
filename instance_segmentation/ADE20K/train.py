import os, sys

from dataset import *

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import model as modellib


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

ADE20K_DIR = "/home/orestisz/data/ADE20K_2016_07_26"

config = Config()
config.display()

# Training dataset
dataset_train = Dataset()
dataset_train.load(ADE20K_DIR, "training")
dataset_train.prepare()

# Validation dataset
dataset_val = Dataset()
dataset_val.load(ADE20K_DIR, "validation")
dataset_val.prepare()

# Create model in training mode
print('creating model..')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
print('loading weights...')
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

## Training

# Train the head branches
print('training heads...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE, 
            epochs=100, 
            layers='heads')

# Fine tune all layers
# print('fine tuning all layers...')
# model.train(dataset_train, dataset_val, 
#             learning_rate=config.LEARNING_RATE,
#             epochs=200, 
#             layers="all")
