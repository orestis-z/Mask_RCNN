import os, sys
import math
import numpy as np
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from object_segmentation.object_config import Config

import utils

class ObjectsConfig(Config):
    NAME = "seg_SceneNet"
    # NAME = "seg_ADE20K"

    MODE = 'RGBD'

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320

    IMAGES_PER_GPU = 2
    # LEARNING_RATE = 0.002
    # LEARNING_RATE = 0.02
    
    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])

class ObjectsDataset(utils.Dataset):
    def load_sceneNet(self, dataset_dir, subset):
        assert(subset == 'training' or subset == 'validation' or subset == 'testing')
        dataset_dir = os.path.join(dataset_dir, subset, '0')

        # Add classes
        self.add_class("seg_sceneNN", 1, "object")
        count = 0
        # Add images
        for i, (root, dirs, files) in enumerate(os.walk(dataset_dir)):
            root_split = root.split('/')
            if count > 0:
                break
            if root_split[-1] == 'photo': # and subset in root_split:
                print('Loading {} data from {}'.format(subset, root_split[-2]))
                for file in files:
                    parentRoot = '/'.join(root.split('/')[:-1])
                    depth_path = os.path.join(parentRoot, 'depth', file[:-4] + '.png')
                    instance_path = os.path.join(parentRoot, 'instance', file[:-4] + '.png')
                    # only add if corresponding mask exists
                    path = os.path.join(root, file)
                    if os.path.isfile(depth_path) and os.path.isfile(instance_path):
                        if (os.stat(path).st_size):
                            im = Image.open(path)
                            width, height = im.size
                            self.add_image(
                                "seg_sceneNN",
                                image_id=i,
                                path=path,
                                depth_path=depth_path,
                                instance_path=instance_path,
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

        unique, unique_inverse = np.unique(img.flatten(), return_inverse=True)
        object_instance_masks = np.reshape(unique_inverse, img.shape)
        instances = np.unique(unique_inverse).tolist()
        instances.remove(0)
        instance_masks = []
        for i, instance in enumerate(instances):
            vfunc = np.vectorize(lambda a: 1 if a == instance else 0)
            instance_masks.append(vfunc(object_instance_masks))
        if not instance_masks:
            raise ValueError("No instances for image {}".format(instance_path))
        masks = np.stack(instance_masks, axis=2)
        class_ids = np.array([1] * len(instances), dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = ObjectsDataset()
    dataset.load_sceneNet('/external_datasets/SceneNet_RGBD', 'validation')
    masks, class_ids = dataset.load_mask(0)
