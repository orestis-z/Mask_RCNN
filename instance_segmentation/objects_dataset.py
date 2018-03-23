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

img_path = 'generated/image'
instance_path = 'generated/instance'

class ObjectsDataset(utils.Dataset):
    def __init__(self, use_generated=False):
        super().__init__()
        self.use_generated = use_generated

    def load_image(self, image_id, depth=True):
        if self.use_generated:
            parent_path = self.image_info[image_id]['parent_path']
            file_name = self.image_info[image_id]['file_name']
            return np.load(os.path.join(parent_path, img_path, file_name + ".npy"))
        image = super().load_image(image_id)
        if depth:
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            rgbd = np.dstack((image, depth))
            return rgbd
        else:
            return image

    def load_mask(self, image_id):
        parent_path = self.image_info[image_id]['parent_path']
        file_name = self.image_info[image_id]['file_name']
        return np.load(os.path.join(parent_path, instance_path, file_name + ".npy"))

    def generate_files(self, dataset_dir, subset, skip=0, path=None, depth=3):
        self.load(dataset_dir, subset, skip=skip)
        self.prepare()
        for image_id in self.image_ids:
            directory = self.image_info[image_id]['parent_path']
            file_name = self.image_info[image_id]['file_name']
            if (path is not None):
                directory = os.path.join(path, *directory.split("/")[-depth:])
            if not os.path.exists(directory + '/generated'):
                os.makedirs(os.path.join(directory, img_path))
                os.makedirs(os.path.join(directory, instance_path))
            print(self.image_info[image_id]['path'] + '  ---->  ' + directory)
            image = self.load_image(image_id)
            np.save(os.path.join(directory, img_path, file_name) + ".npy", image)
            mask = self.load_mask(image_id)
            np.save(os.path.join(directory, instance_path, file_name + ".npy"), image)
