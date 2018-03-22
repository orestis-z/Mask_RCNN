import os, sys
import numpy as np
import scipy
from scipy.io import loadmat
from PIL import Image

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset


class Config(ObjectsConfig):
    NAME = "seg_ADE20K"

class Dataset(ObjectsDataset):
    def load(self, dataset_dir, subset):
        assert(subset == 'training' or subset == 'validation')
        index = loadmat(os.path.join(dataset_dir, 'index_ade20k.mat'))['index']

        # All images
        image_ids = range(len(index['folder'][0][0][0]))

        # Add classes
        self.add_class("seg_ADE20K", 1, "object")

        # Add images
        folders = index['folder'][0][0][0]
        for i in image_ids:
            folder = '/'.join(folders[i][0].split('/')[1:])
            if subset in folder.split('/'):
                file_name = index['filename'][0][0][0][i][0]
                path = os.path.join(dataset_dir, folder, file_name)
                im = Image.open(path)
                width, height = im.size
                self.add_image(
                    "seg_ADE20K",
                    image_id=i,
                    path=path,
                    width=width,
                    height=height)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        seg_path = image_info['path'][:-4] + '_seg.png'
        seg = scipy.misc.imread(seg_path)

        # R = seg[:, :, 0]
        # G = seg[:, :, 1]
        B = seg[:, :, 2]

        # port to python from matlab script:
        # http://groups.csail.mit.edu/vision/datasets/ADE20K/code/loadAde20K.m
        # object_class_masks = (R.astype(np.uint16) / 10) * 256 + G.astype(np.uint16)
        instances = np.unique(B.flatten())
        instances = instances.tolist()
        if 0 in instances:
            instances.remove(0)
        n_instances = len(instances)
        masks = np.zeros((img.shape[0], img.shape[1], n_instances))
        for i, instance in enumerate(instances):
            masks[:, :, i] = (B == instance).astype(np.uint8)
        if not n_instances:
            raise ValueError("No instances for image {}".format(instance_path))

        class_ids = np.array([1] * n_instances, dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load('/home/orestisz/data/ADE20K_2016_07_26', 'validation')
    masks, class_ids = dataset.load_mask(0)
