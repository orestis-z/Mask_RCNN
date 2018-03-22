import os, sys
import math
import numpy as np
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils


class ObjectsDataset(utils.Dataset):
    def __init__(self, use_generated=False):
        super().__init__()
        self.use_generated = use_generated

    def load_image(self, image_id, depth=True):
        if self.use_generated:
            return np.load(self.image_info[image_id]['path'][:-4] + "_generated.npy")
        image = super().load_image(image_id)
        if depth:
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            rgbd = np.dstack((image, depth))
            return rgbd
        else:
            return image

    def load_mask(self, image_id):
        return np.load(self.image_info[image_id]['mask_path'][:-4] + "_generated.npy")

    def generate_files(self, dataset_dir, subset, skip=0):
        load(dataset_dir, subset, skip=skip)
        for image_id in self.image_ids:
            image = self.load_image()
            np.save(self.image_info[image_id]['path'][:-4] + "_generated.npy", image)
            mask = self.load_mask(image_id)
            np.save(self.image_info[image_id]['mask_path'][:-4] + "_generated.npy", image)
