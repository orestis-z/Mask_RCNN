import os, sys
import math
import numpy as np
import cv2
import skimage.io
import random


# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils import Dataset

img_path = 'generated/image'
instance_path = 'generated/instance'


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if img_min == img_max:
        return np.ones(img.shape) * 127.5
    return np.clip((img - img_min) / (img_max - img_min), 0, 1) * 255

class ObjectsDataset(Dataset):
    def __init__(self, use_generated=False):
        super().__init__()
        self.use_generated = use_generated

    def load_image(self, image_id, mode="RGB"):
        if self.use_generated:
            parent_path = self.image_info[image_id]['parent_path']
            file_name = self.image_info[image_id]['file_name']
            return np.load(os.path.join(parent_path, img_path, file_name + ".npy"))
        image = super().load_image(image_id)
        if mode == "RGBD":
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            depth = normalize(depth)
            rgbd = np.dstack((image, depth))
            ret = rgbd
        else:
            ret = image
        return ret

    def load_mask(self, image_id):
        parent_path = self.image_info[image_id]['parent_path']
        file_name = self.image_info[image_id]['file_name']
        masks = np.load(os.path.join(parent_path, instance_path, file_name + ".npy"))
        return masks, np.array([1] * masks.shape[2], dtype=np.int32) 

    # attempt to generate files in format best suitable for direct loading of mask instances and therefore skipping computetionally expensive conversion
    def generate_files(self, dataset_dir, subset, path=None, depth=3):
        self.load(dataset_dir, subset)
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
