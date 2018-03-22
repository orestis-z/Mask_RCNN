import os, sys
import math
import numpy as np
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.object_config import Config

import utils

class ObjectsConfig(Config):
    NAME = "seg_sceneNet"
    # NAME = "seg_ADE20K"

    MODE = 'RGBD'
    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320

    IMAGES_PER_GPU = 4
    LEARNING_RATE = 0.002 / 10
    
    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])

class ObjectsDataset(utils.Dataset):
    CLASSES = [ (0,'Unknown'),
                (1,'Bed'),
                (2,'Books'),
                (3,'Ceiling'),
                (4,'Chair'),
                (5,'Floor'),
                (6,'Furniture'),
                (7,'Objects'),
                (8,'Picture'),
                (9,'Sofa'),
                (10,'Table'),
                (11,'TV'),
                (12,'Wall'),
                (13,'Window')]

    # def __init__(self, class_map=None):
    #     super().__init__()
    #     self.class_info = []

    def load(self, dataset_dir, subset, skip=19):
        assert(subset == 'training' or subset == 'validation' or subset == 'testing')
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add classes
        # for cls in self.CLASSES:
        #     self.add_class("seg_sceneNet", cls[0], cls[1])
        # for i in range(1000):
        #     self.add_class("seg_sceneNet", i, str(i))

        self.add_class("seg_sceneNet", 1, 'object')

        count = 0
        exclude = set(['depth', 'instance'])
        # Add images
        for i, (root, dirs, files) in enumerate(os.walk(dataset_dir, topdown=True)):
            dirs[:] = [d for d in dirs if d not in exclude]
            root_split = root.split('/')
            if root_split[-1] == 'photo': # and subset in root_split:
                print('Loading {} data from {}, {}'.format(subset, root_split[-3], root_split[-2]))
                for j, file in enumerate(files):
                    if j % (skip + 1) == 0:
                        parentRoot = '/'.join(root.split('/')[:-1])
                        depth_path = os.path.join(parentRoot, 'depth', file[:-4] + '.png')
                        instance_path = os.path.join(parentRoot, 'instance', file[:-4] + '.png')
                        path = os.path.join(root, file)
#                         im = Image.open(path)
#                         width, height = im.size
                        width, height = (320, 240)
                        self.add_image(
                            "seg_sceneNet",
                            image_id=i,
                            path=path,
                            depth_path=depth_path,
                            instance_path=instance_path,
                            width=width,
                            height=height)
                        count += 1
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, depth=True):
        """Load the specified image and return a [H,W,3+1] Numpy array.
        """
        # Load image & depth
        image = super(ObjectsDataset, self).load_image(image_id)
        if depth:
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
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
        instance_path = self.image_info[image_id]['instance_path']
        img = np.asarray(Image.open(instance_path))

        instances = np.unique(img.flatten())
        # instances = instances.tolist()
        # instances.remove(0)
        n_instances = len(instances)
        masks = np.zeros((img.shape[0], img.shape[1], n_instances))
        for i, instance in enumerate(instances):
            masks[:, :, i] = (img == instance).astype(np.uint8)
        if not n_instances:
            raise ValueError("No instances for image {}".format(instance_path))

        class_ids = np.array([1] * n_instances, dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = ObjectsDataset()
    dataset.load_sceneNet('/external_datasets/SceneNet_RGBD', 'validation', skip=299)
    masks, class_ids = dataset.load_mask(0)
