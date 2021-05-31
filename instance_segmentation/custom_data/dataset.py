import os
import sys

import cv2
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath('../..')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset


n_images = 19

PROCESS_DEPTH = 'inpaint'  # None, inpaint, open

# kernel for opening
s = 4
kernel = np.ones((s, s), np.uint8)
kernel[0, 0] = kernel[s - 1, s - 1] = kernel[0, s - 1] = kernel[s - 1, 0] = 0


class Config(ObjectsConfig):
    NAME = 'custom_data'

    MODE = 'RGBD'
    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 255.0 / 2])


class Dataset(ObjectsDataset):
    subset = 'validation'

    WIDTH = 640
    HEIGHT = 480

    def load(self, dataset_dir):
        self.add_class('custom_data', 1, 'object')

        # Add images
        for i in range(1, n_images + 1):
            self.add_image(
                'custom_data',
                image_id=i,
                path=os.path.join(dataset_dir, 'RGB{}.jpg'.format(i)),
                depth_path=os.path.join(dataset_dir, 'depth{}.png'.format(i)),
                width=self.WIDTH,
                height=self.HEIGHT)

    def load_image(self, image_id, mode='RGBD'):
        ret = super().load_image(image_id, mode)
        if mode == 'RGBD':
            depth_raw = ret[:, :, 3]
            mask = (depth_raw == 0).astype(np.uint8)

            if PROCESS_DEPTH == 'inpaint':
                depth = cv2.inpaint(
                    depth_raw.astype(
                        np.uint8), mask, 3, cv2.INPAINT_NS)
            elif PROCESS_DEPTH == 'open':
                opening = (1 - cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                           kernel, iterations=5)) * mask
                depth = cv2.inpaint(
                    depth_raw.astype(
                        np.uint8), opening.astype(
                        np.uint8), 3, cv2.INPAINT_NS)
            else:
                depth = depth_raw

            ret = np.dstack((ret[:, :, 0:3], depth, depth_raw, mask))
        return ret
