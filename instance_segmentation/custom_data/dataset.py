import os, sys
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset


n_images = 19

class Config(ObjectsConfig):
    NAME = "custom_data"

    MODE = 'RGBDE'
    BACKBONE = 'resnet50'

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7, 255.0 / 100])

class Dataset(ObjectsDataset):
    def load(self, dataset_dir):
        self.add_class("custom_data", 1, 'object')

        # Add images
        for i in range(1, n_images + 1):
            width, height = (640, 480)
            self.add_image(
                "custom_data",
                image_id=i,
                path=os.path.join(dataset_dir, 'RGB{}.jpg'.format(i)),
                depth_path=os.path.join(dataset_dir, 'depth{}.png'.format(i)),
                width=width,
                height=height)
