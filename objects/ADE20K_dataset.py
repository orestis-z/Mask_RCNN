"""
Mask R-CNN
Configurations and data loading code for the synthetic Objects dataset.
This is a duplicate of the code in the noteobook train_objects.ipynb for easy
import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os, sys
import math
import random
import numpy as np
import cv2
from random import randint
import scipy
from scipy.io import loadmat
from PIL import Image

from object_config import Config

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import utils

class ObjectsConfig(Config):
    NAME = "objects"
    # """Configuration for training on the toy objects dataset.
    # Derives from the base Config class and overrides values specific
    # to the toy objects dataset.
    # """
    # # Give the configuration a recognizable name
    # NAME = "objects"

    # # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    # GPU_COUNT = 4
    # IMAGES_PER_GPU = 8

    # # Number of classes (including background)
    # NUM_CLASSES = 1 + 1  # background + 3 objects

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


class ObjectsDataset(utils.Dataset):
    """Generates the objects synthetic dataset. The dataset consists of simple
    objects (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_ADE20K(self, dataset_dir, subset):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        # Path
        # image_dir = os.path.join(dataset_dir, "train2014" if subset == "train"
        #                          else "val2014")
        assert(subset == 'training' or subset == 'validation')

        index = loadmat(os.path.join(dataset_dir, 'index_ade20k.mat'))['index']

        # Create COCO object
        # json_path_dict = {
        #     "train": "annotations/instances_train2014.json",
        #     "val": "annotations/instances_val2014.json",
        #     "minival": "annotations/instances_minival2014.json",
        #     "val35k": "annotations/instances_valminusminival2014.json",
        # }
        # coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))

        # All images
        image_ids = range(len(index['folder'][0][0][0]))

        # Add classes
        self.add_class("objects", 1, "object")

        # Add images
        folders = index['folder'][0][0][0]
        for i in image_ids:
            folder = '/'.join(folders[i][0].split('/')[1:])
            if subset in folder.split('/'):
                file_name = index['filename'][0][0][0][i][0]
                path = os.path.join(dataset_dir, folder, file_name)
                im = Image.open(path)
                width, height = im.size
                self.add_image(
                    "objects", image_id=i,
                    path=path,
                    width=width,
                    height=height)
                    # annotations=)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        seg_path = image_info['path'][:-4] + '_seg.png'
        seg = scipy.misc.imread(seg_path)
        # R = seg[:, :, 0]
        # G = seg[:, :, 1]
        B = seg[:, :, 2]

        # object_class_masks = (R.astype(np.uint16) / 10) * 256 + G.astype(np.uint16)
        unique, unique_inverse = np.unique(B.flatten(), return_inverse=True)
        object_instance_masks = np.reshape(unique_inverse, B.shape)
        instances = np.unique(unique_inverse).tolist()
        instances.remove(0)
        instance_count = len(instances)
        instance_masks = []
        for i, instance in enumerate(instances):
            vfunc = np.vectorize(lambda a: 1 if a == instance else 0)
            instance_masks.append(vfunc(object_instance_masks))

        masks = np.stack(instance_masks, axis=2)
        class_ids = np.array([1] * instance_count, dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = ObjectsDataset()
    dataset.load_AD20K('/home/orestisz/data/ADE20K_2016_07_26', 'validation')
    masks, class_ids = dataset.load_mask(0)
