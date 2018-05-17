"""
checkout toolbox at https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
to understand how labels are loded
"""

import os, sys
import numpy as np
import cv2
import skimage.io
import random

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset, normalize
from data.names import names


BINARY_CLASS = True
# BINARY_CLASS = False
if BINARY_CLASS:
    NAME = "NYU_Depth_V2_scenenet_rgb_all_layers_2"
else:
    NAME = "NYU_Depth_V2_scenenet_rgb_all_layers_2_classes"

EXCLUDE = ['floor', 'wall', 'ceiling'] # exclude stuff (include only well-localized objects)

class Config(ObjectsConfig):
    NAME = NAME

    MODE = 'RGBD'
    # MODE = 'RGB'
    BACKBONE = 'resnet50'
    # BACKBONE = 'resnet101'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001

    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 255 / 2]) # 1220.7 / 1000, 255.0 / 100])

    def __init__(self):
        super().__init__()
        if not BINARY_CLASS:
            self.NUM_CLASSES = len(names) + 1 - len(EXCLUDE)


class Dataset(ObjectsDataset):
    WIDTH = 640
    HEIGHT = 480

    def load(self, dataset_dir, subset, skip=0):
        self.subset = subset
        assert(subset == 'training' or subset == 'validation')

        dataset_dir = os.path.join(dataset_dir, subset)

        if BINARY_CLASS:
            self.add_class(NAME, 1, 'object')
        else:
            for idx, name in enumerate(names):
                self.add_class(NAME, idx + 1, name)

        count = 0
        exclude = set(['depths', 'instances', 'labels'])
        # Add images
        for root, dirs, files in os.walk(dataset_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude]
            root_split = root.split('/')
            if root_split[-1] == 'images':
                print('Loading {} data from {}, {}'.format(subset, root_split[-3], root_split[-2]))
                for j, file in enumerate(files):
                    if j % (skip + 1) == 0:
                        parent_path = '/'.join(root.split('/')[:-1])
                        depth_path = os.path.join(parent_path, 'depths', file)
                        instances_path = os.path.join(parent_path, 'instances', file)
                        labels_path = os.path.join(parent_path, 'labels', file)
                        path = os.path.join(root, file)
                        self.add_image(
                            NAME,
                            image_id=count,
                            path=path,
                            depth_path=depth_path,
                            instances_path=instances_path,
                            labels_path=labels_path,
                            parent_path=parent_path,
                            width=self.WIDTH,
                            height=self.HEIGHT)
                        count += 1
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, mode="RGBD"):
        if self.use_generated:
            parent_path = self.image_info[image_id]['parent_path']
            file_name = self.image_info[image_id]['file_name']
            return np.load(os.path.join(parent_path, img_path, file_name + ".npy"))
        image = np.load(self.image_info[image_id]['path']).T
        if mode == "RGBD":
            depth = np.load(self.image_info[image_id]['depth_path']).T
            depth = normalize(depth)
            rgbd = np.dstack((image, depth))
            ret = rgbd
        else:
            ret = image
        return ret

    def to_mask(inst_img, labels_img, label, instance):
        return np.bitwise_and(inst_img == instance, labels_img == label)

    to_mask_v = np.vectorize(to_mask, signature='(n,m),(n,m),(k),(k)->(n,m)')

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        instances_path = image_info['instances_path']
        labels_path = image_info['labels_path']
        instances_img = np.load(instances_path).T
        labels_img = np.load(labels_path).T

        instances = instances_img.flatten().astype(np.uint16)

        labels = labels_img.flatten()
        unique_instances = np.unique(np.stack((labels, instances)), axis=1)
        labels = unique_instances[0]
        instances = unique_instances[1]

        instances = instances.tolist()
        labels = labels.tolist()
        if 0 in labels:
            x = labels.index(0)
            del labels[x]
            del instances[x]
        for excl in EXCLUDE:
            idx = names.index(excl) + 1
            while idx in labels:
                x = labels.index(idx)
                del labels[x]
                del instances[x]

        unique_instances = np.stack((labels, instances))

        n_instances = unique_instances.shape[1]
        instances = np.repeat(np.expand_dims(instances_img, axis=2), n_instances, axis=2)
        labels = np.repeat(np.expand_dims(labels_img, axis=2), n_instances, axis=2)
        masks = self.to_mask_v(instances, labels, unique_instances[0], unique_instances[1])
        if not n_instances:
            raise ValueError("No instances for image {}".format(mask_path))

        if BINARY_CLASS:
            class_ids = np.array([1] * n_instances, dtype=np.int32)
        else:
            class_ids = np.array(unique_instances[0], dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load('/home/orestisz/repositories/Mask_RCNN/instance_segmentation/NYU-Depth_V2/data', 'validation')
    masks, class_ids = dataset.load_mask(0)

