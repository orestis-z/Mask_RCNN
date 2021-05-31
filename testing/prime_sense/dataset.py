import os
import sys

import numpy as np
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath('../..')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils
from instance_segmentation.object_config import Config


class ObjectsConfig(Config):
    NAME = 'prime_sense'
    # NAME = "seg_ADE20K"

    MODE = 'RGBD'

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640

    # IMAGES_PER_GPU = 2
    # LEARNING_RATE = 0.02

    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])


class ObjectsDataset(utils.Dataset):
    def load(self, dataset_dir, skip=19):
        # Add classes
        self.add_class('prime_sense', 1, 'object')
        count = 0
        # Add images
        for i, (root, dirs, files) in enumerate(os.walk(dataset_dir)):
            root_split = root.split('/')
            if root_split[-1] == 'image':  # and subset in root_split:
                for j, file in enumerate(files):
                    if j % (skip + 1) == 0:
                        parentRoot = '/'.join(root.split('/')[:-1])
                        depth_path = os.path.join(parentRoot, 'depth', file)
                        # only add if corresponding mask exists
                        path = os.path.join(root, file)
                        if os.path.isfile(depth_path):
                            if (os.stat(path).st_size):
                                im = Image.open(path)
                                width, height = im.size
                                self.add_image(
                                    'prime_sense',
                                    image_id=i,
                                    path=path,
                                    depth_path=depth_path,
                                    width=width,
                                    height=height)
                                count += 1
                        else:
                            print(
                                'Warning: No depth or mask found for ' + path)
        print('added {} images'.format(count))

    def load_image(self, image_id, depth=True):
        """Load the specified image and return a [H,W,3+1] Numpy array."""
        # Load image & depth
        image = super(ObjectsDataset, self).load_image(image_id)
        if depth:
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            rgbd = np.dstack((image, depth))
            return rgbd
        else:
            return image


if __name__ == '__main__':
    dataset = ObjectsDataset()
    dataset.load('/home/orestisz/data/ADE20K_2016_07_26', 'validation')
    masks, class_ids = dataset.load_mask(0)
