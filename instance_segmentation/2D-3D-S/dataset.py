import os, sys
import math
import numpy as np
import skimage.io
import cv2
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.object_config import Config

import utils

class ObjectsConfig(Config):
    NAME = "2D-3D-S"

    MODE = 'RGBD'

    IMAGE_MIN_DIM = 640
    IMAGE_MAX_DIM = 640

    # IMAGES_PER_GPU = 2
    # LEARNING_RATE = 0.02

    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])

class ObjectsDataset(utils.Dataset):
    def load(self, dataset_dir, subset, skip=0):
        assert(subset == 'training' or subset == 'validation' or subset == 'testing')
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add classes
        self.add_class("2D-3D-S", 1, "object")

        count = 0
        exclude = set(['3d', 'pano', 'raw', 'sensor', 'semantic_pretty', 'normal', 'pose', 'depth', 'semantic'])
        # Add images
        for i, (root, dirs, files) in enumerate(os.walk(dataset_dir, topdown=True)):
            dirs[:] = [d for d in dirs if d not in exclude]
            root_split = root.split('/')
            if root_split[-2] == 'data' and root_split[-1] == 'rgb': # and subset in root_split:
                print(root_split[-3])
                for j, file in enumerate(files):
                    if j % (skip + 1) == 0:
                        parentRoot = '/'.join(root.split('/')[:-1])
                        depth_path = os.path.join(parentRoot, 'depth', file[:-7] + 'depth.png')
                        mask_path = os.path.join(parentRoot, 'semantic', file[:-7] + 'semantic.png')
                        # only add if corresponding mask exists
                        path = os.path.join(root, file)
                        if os.path.isfile(depth_path) and os.path.isfile(mask_path):
                            # if (os.stat(path).st_size):
                            width, height = (1080, 1080)
                            self.add_image(
                                "2D-3D-S",
                                image_id=i,
                                path=path,
                                depth_path=depth_path,
                                mask_path=mask_path,
                                width=width,
                                height=height)
                            count += 1
                        else:
                            print('Warning: No depth or mask found for ' + path)
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, depth=True):
        """Load the specified image and return a [H,W,3+1] Numpy array.
        """
        # Load image & depth
        image = super(ObjectsDataset, self).load_image(image_id)
        if depth:
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            depth = np.clip(depth, 0, 5500)
            rgbd = np.dstack((image, depth))
            return rgbd
        else:
            return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        mask_path = self.image_info[image_id]['mask_path']
        img = cv2.imread(mask_path, -1)

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img = R * 256 * 256 + G * 256 + B

        instances, unique_inverse = np.unique(img.flatten(), return_inverse=True)
        object_instance_mask_idx = np.reshape(unique_inverse, img.shape)
        instance_idx = np.unique(unique_inverse).tolist()
        instance_idx.remove(0)
        instance_masks = []
        for i, instance_i in enumerate(instance_idx):
            vfunc = np.vectorize(lambda a: 1 if a == instance_i else 0)
            instance_masks.append(vfunc(object_instance_mask_idx))
        if not instance_masks:
            raise ValueError("No instances for image {}".format(instance_path))
        masks = np.stack(instance_masks, axis=2)
        # class_ids = np.array(instances, dtype=np.int32)
        class_ids = np.array([1] * len(instance_idx), dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = ObjectsDataset()
    dataset.load('/external_datasets/2D-3D-S', 'testing', skip=199)
    masks, class_ids = dataset.load_mask(0)
