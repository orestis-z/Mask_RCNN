import os, sys
import numpy as np
import scipy
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from object_segmentation.object_config import Config

import utils

class ObjectsConfig(Config):
    NAME = "seg_sceneNN"

    MODE = 'RGBD'

    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 100.0])

class ObjectsDataset(utils.Dataset):
    def load_sceneNN(self, dataset_dir, subset):
        assert(subset == 'training' or subset == 'validation' or subset == 'testing')
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add classes
        self.add_class("seg_sceneNN", 1, "object")
        # Add images
        for i, (root, dirs, files) in enumerate(os.walk(dataset_dir)):
            root_split = root.split('/')
            if root_split[-1] == 'image': # and subset in root_split:
                for file in files:
                    parentRoot = '/'.join(root.split('/')[:-1])
                    depth_path = os.path.join(parentRoot, 'depth', 'depth' + file[5:])
                    mask_path = os.path.join(parentRoot, 'mask', 'mask_' + file)
                    # only add if corresponding mask exists
                    if os.path.isfile(depth_path) and os.path.isfile(mask_path):
                        path = os.path.join(root, file)
                        if (os.stat(path).st_size):
                            im = Image.open(path)
                            width, height = im.size
                            self.add_image(
                                "seg_sceneNN",
                                image_id=i,
                                path=path,
                                depth_path=depth_path,
                                mask_path=mask_path,
                                width=width,
                                height=height)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3+1] Numpy array.
        """
        # Load image & depth
        image = super(ObjectsDataset, self).load_image(image_id)
        depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
        rgbd = np.dstack((image, depth))

        return rgbd

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        img = scipy.misc.imread(image_info['mask_path'], mode='RGBA')

        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]
        A = img[:, :, 3]

        # port to python from cpp script:
        # https://github.com/scenenn/shrec17/blob/master/mask_from_label/mask_from_label.cpp
        seg = np.bitwise_or(np.bitwise_or(np.bitwise_or(
                np.left_shift(R, 24),
                np.left_shift(G, 16)),
                np.left_shift(B, 8)),
                A)

        # object_class_masks = (R.astype(np.uint16) / 10) * 256 + G.astype(np.uint16)
        unique, unique_inverse = np.unique(seg.flatten(), return_inverse=True)
        object_instance_masks = np.reshape(unique_inverse, seg.shape)
        instances = np.unique(unique_inverse).tolist()
        instances.remove(0)
        instance_masks = []
        for i, instance in enumerate(instances):
            vfunc = np.vectorize(lambda a: 1 if a == instance else 0)
            instance_masks.append(vfunc(object_instance_masks))

        masks = np.stack(instance_masks, axis=2)
        class_ids = np.array([1] * len(instances), dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = ObjectsDataset()
    dataset.load_AD20K('/home/orestisz/data/ADE20K_2016_07_26', 'validation')
    masks, class_ids = dataset.load_mask(0)
