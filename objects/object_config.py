import os, sys

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from config import Config as _Config

class Config(_Config):
    """Configuration for training on the toy objects dataset.
    Derives from the base Config class and overrides values specific
    to the toy objects dataset.
    """
    # Give the configuration a recognizable name
    NAME = "objects"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    # GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 object

    # # Use small images for faster training. Set the limits of the small side
    # # the large side, and that determines the image shape.
    # IMAGE_MIN_DIM = 128
    # IMAGE_MAX_DIM = 128

    # # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # # Reduce training ROIs per image because the images are small and have
    # # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # # Use a small epoch since the data is simple
    # STEPS_PER_EPOCH = 100

    # # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5
