import os
import sys

from dataset import *

# Root directory of the project
ROOT_DIR = os.path.abspath('../..')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import model as modellib


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')

# Path to ADE20K trained weights
# WEIGHTS_PATH = os.path.join(ROOT_DIR, "logs/2d-3d-s20180314T1711/mask_rcnn_2d-3d-s_0167.h5")

WEIGHTS_PATH = '/data/orestisz/logs/2d_3d_s20180319T0149/mask_rcnn_2d_3d_s_0016.h5'

DATASET_DIR = '/external_datasets/2D-3D-S'

config = Config()
config.display()

# Create model in training mode
print('creating model..')
model = modellib.MaskRCNN(mode='training', config=config,
                          model_dir=MODEL_DIR)

# # Which weights to start with?
init_with = 'custom'  # custom or last

print('loading weights...')
if init_with == 'custom':
    model.load_weights(WEIGHTS_PATH, by_name=True)
elif init_with == 'last':
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

# Training

# Training dataset
dataset_train = Dataset()
dataset_train.load(DATASET_DIR, 'training')
dataset_train.prepare()

# Validation dataset
dataset_val = Dataset()
dataset_val.load(DATASET_DIR, 'testing')
dataset_val.prepare()

# Fine tune all layers
print('fine tuning all layers...')
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1000,
            layers='all')
            # layers='|'.join(exclude))
