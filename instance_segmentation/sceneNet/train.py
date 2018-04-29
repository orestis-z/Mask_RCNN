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
SCENENN_MODEL_PATH = os.path.join(ROOT_DIR, "logs/mask_rcnn_seg_scenenn_0101.h5")
SCENENET_MODEL_PATH = os.path.join(MODEL_DIR, "seg_scenenet20171121T1912/mask_rcnn_seg_scenenet_0692.h5")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_PATH = COCO_MODEL_PATH

config = Config()
config.display()

# Training dataset
dataset_train = Dataset()
dataset_train.load("training")
dataset_train.prepare()

# Validation dataset
dataset_val = Dataset()
dataset_val.load("validation")
dataset_val.prepare()

# Create model in training mode
print('creating model..')
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

exclude = ["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"]

# # Which weights to start with?
init_with = "coco"  # scenenn, last, imagenet

print('loading {} weights...'.format(init_with))
if init_with == "last":
    print(model.find_last()[1])
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)
elif init_with == "scenenn":
    model.load_weights(SCENENN_MODEL_PATH, by_name=True,
                       exclude=exclude)
elif init_with == "sceneNet":
    model.load_weights(SCENENET_MODEL_PATH, by_name=True,
                       exclude=exclude)
elif init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=exclude)
elif init_with == "rgb_to_rgbd":
    model.load_weights(MODEL_PATH, by_name=True, exclude=exclude)

    # get weights from first convolution
    conv1 = model.keras_model.get_layer("conv1")
    kernel_rgb, bias = conv1.get_weights()

    # reload model in rgb-d mode
    config_rgbd = Config()
    config_rgbd.MODE = "RGBD"
    config_rgbd.MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 255 / 2])
    config_rgbd.IMAGE_SHAPE = np.array([config_rgb.IMAGE_MAX_DIM, config_rgb.IMAGE_MAX_DIM, 4])
    model_rgbd = modellib.MaskRCNN(mode="inference", config=config_rgbd, model_dir=MODEL_DIR)
    model_rgbd.load_weights(MODEL_PATH, by_name=True, exclude=["conv1"])

    # set weights on first convolution from rgb model
    conv1 = model_rgbd.keras_model.get_layer("conv1")
    kernel_rgbd = np.concatenate((kernel_rgb, np.mean(kernel_rgb, keepdims=True, axis=2)), axis=2)
    conv1.set_weights([kernel_rgbd, bias])

## Training

print('tuning heads...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=3,
            layers="heads")
print('tuning layers 5+...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=7,
            layers="5+")
print('tuning layers 4+...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=12,
            layers="4+")
print('tuning layers 3+...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=18,
            layers="3+")
print('tuning all layers...')
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE,
            epochs=1000,
            layers="all")
