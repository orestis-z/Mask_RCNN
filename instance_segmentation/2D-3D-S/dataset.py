import os, sys
import numpy as np
import skimage.io
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset, normalize


NAME = "2D_3D_S"

class Config(ObjectsConfig):
    NAME = NAME

    MODE = 'RGBD'
    BACKBONE = 'resnet50'

    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 448

    IMAGES_PER_GPU = 2
    LEARNING_RATE = 0.002

    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 127.5])

class Dataset(ObjectsDataset):
    WIDTH = 1080
    HEIGHT = 1080

    def load(self, dataset_dir, subset, skip=0):
        assert(subset == 'training' or subset == 'validation' or subset == 'testing')
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add binary class
        self.add_class(NAME, 1, "object")

        count = 0
        exclude = set(['3d', 'pano', 'raw', 'sensor', 'semantic_pretty', 'normal', 'pose', 'depth', 'semantic'])
        # Add images
        for root, dirs, files in os.walk(dataset_dir, topdown=True):
            dirs[:] = [d for d in dirs if d not in exclude]
            root_split = root.split('/')
            if root_split[-2] == 'data' and root_split[-1] == 'rgb': # and subset in root_split:
                print(root_split[-3])
                for j, file in enumerate(files):
                    if file[-4:] == '.png' and j % (skip + 1) == 0:
                        parent = '/'.join(root.split('/')[:-1])
                        file_name = file[:-7]
                        depth_path = os.path.join(parent, 'depth', file_name + 'depth.png')
                        mask_path = os.path.join(parent, 'semantic', file_name + 'semantic.png')
                        # only add if corresponding mask exists
                        path = os.path.join(root, file)
                        if os.path.isfile(depth_path) and os.path.isfile(mask_path):
                            self.add_image(
                                NAME,
                                image_id=count,
                                path=path,
                                depth_path=depth_path,
                                mask_path=mask_path,
                                width=self.WIDTH,
                                height=self.HEIGHT)
                            count += 1
                        else:
                            print('Warning: No depth or mask found for ' + path)
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, mode="RGBD"):
        """Load the specified image and return a [H,W,3+1] Numpy array.
        """
        # Load image & depth
        image = super(Dataset, self).load_image(image_id)
        if mode == "RGBD":
            depth = skimage.io.imread(self.image_info[image_id]['depth_path'])
            depth = normalize(np.clip(depth, 0, 5500)) # clip depth to max 5.5m
            rgbd = np.dstack((image, depth))
            return rgbd
        else:
            return image

    def to_mask(img, instance):
        return (img == instance).astype(np.uint8)

    # vectorize since this was slow in serial execution
    to_mask_v = np.vectorize(to_mask, signature='(n,m),(k)->(n,m)')

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        mask_path = self.image_info[image_id]['mask_path']
        img = cv2.imread(mask_path, -1)

        # https://github.com/alexsax/2D-3D-Semantics/blob/master/assets/utils.py
        R = img[:, :, 0]
        G = img[:, :, 1]
        B = img[:, :, 2]

        img = R * 256 * 256 + G * 256 + B

        instances = np.unique(img.flatten())
        # instances = instances.tolist()
        # if 0 in instances:
        #     instances.remove(0)
        n_instances = len(instances)
        masks = np.repeat(np.expand_dims(img, axis=2), n_instances, axis=2) # bottleneck code
        masks = self.to_mask_v(masks, instances)
        if not n_instances:
            raise ValueError("No instances for image {}".format(instance_path))

        class_ids = np.array([1] * n_instances, dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load('/external_datasets/2D-3D-S', 'testing', skip=999)
    masks, class_ids = dataset.load_mask(0)
