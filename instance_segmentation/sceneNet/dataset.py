import os, sys
import math
import numpy as np
import skimage.io
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset


class Config(ObjectsConfig):
    NAME = "seg_sceneNet"
    # NAME = "seg_ADE20K"

    MODE = 'RGBD'
    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 320

    IMAGES_PER_GPU = 6
    LEARNING_RATE = 0.02
    
    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])

class Dataset(ObjectsDataset):
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
                        mask_path = os.path.join(parentRoot, 'instance', file[:-4] + '.png')
                        path = os.path.join(root, file)
#                         im = Image.open(path)
#                         width, height = im.size
                        width, height = (320, 240)
                        self.add_image(
                            "seg_sceneNet",
                            image_id=i,
                            path=path,
                            depth_path=depth_path,
                            mask_path=mask_path,
                            width=width,
                            height=height)
                        count += 1
        print('added {} images for {}'.format(count, subset))

    def to_mask(mask_img, instance):
        return (mask_img == instance)

    to_mask_v = np.vectorize(to_mask, signature='(n,m),(k)->(n,m)')

    def load_mask(self, image_id):
        if self.use_generated:
            return super().load_mask(image_id)

        mask_path = self.image_info[image_id]['mask_path']
        mask_img = np.asarray(Image.open(mask_path))

        instances = np.unique(mask_img.flatten())
        # instances = instances.tolist()
        # instances.remove(0)
        n_instances = len(instances)
        masks = np.repeat(np.expand_dims(mask_img, axis=2), n_instances, axis=2) # bottleneck code
        masks = self.to_mask_v(masks, instances)
        if not n_instances:
            raise ValueError("No instances for image {}".format(mask_path))

        class_ids = np.array([1] * n_instances, dtype=np.int32)
        # class_ids = np.array(instances, dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = Dataset()
    dataset.generate_files('/external_datasets/SceneNet_RGBD', 'validation')
    dataset.generate_files('/external_datasets/SceneNet_RGBD', 'testing')
