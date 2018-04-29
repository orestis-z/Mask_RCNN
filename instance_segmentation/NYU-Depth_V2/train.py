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
MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# MODEL_PATH = os.path.join(MODEL_DIR, "nyu_depth_v2_scenenet20171121T1912/mask_rcnn_nyu_depth_v2_scenenet_0704.h5")

# MODEL_PATH = os.path.join(MODEL_DIR, "nyu_depth_v2_scenenet20171121T1912/mask_rcnn_nyu_depth_v2_scenenet_0704.h5")

DATASET_DIR = "data"

config = Config()
config.display()

# Training dataset
dataset_train = Dataset()
dataset_train.load(DATASET_DIR, "training")
dataset_train.prepare()

# Validation dataset
dataset_val = Dataset()
dataset_val.load(DATASET_DIR, "validation")
dataset_val.prepare()

# Create model in training mode
print('creating model..')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]

# # Which weights to start with?
init_with = "custom"  # scenenn, last, imagenet

print('loading weights...')
if init_with == "custom":
    model.load_weights(MODEL_PATH, by_name=True, exclude=exclude)
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

## Training

# Fine tune all layers
print('fine tuning all layers...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=1000,
            layers='heads')
