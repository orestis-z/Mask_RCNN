"""Mask R-CNN Configurations and data loading code for the synthetic Objects
dataset. This is a duplicate of the code in the noteobook train_objects.ipynb
for easy import into other notebooks, such as inspect_model.ipynb.

Copyright (c) 2017 Matterport, Inc. Licensed under the MIT License (see
LICENSE for details) Written by Waleed Abdulla
"""

import os
import sys

import numpy as np
from pycocotools.coco import COCO

# Root directory of the project
ROOT_DIR = os.path.abspath('../..')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from coco import CocoDataset
from instance_segmentation.objects_config import ObjectsConfig


Config = ObjectsConfig


class Dataset(CocoDataset):
    """Generates the objects synthetic dataset.

    The dataset consists of simple objects (triangles, squares, circles)
    placed randomly on a blank surface. The images are generated on the
    fly. No file access required.
    """

    def load(self, dataset_dir, subset, class_ids=None):
        """Load a subset of the COCO dataset.

        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        """
        # Path
        image_dir = os.path.join(
            dataset_dir,
            'train2014' if subset == 'train' else 'val2014')

        # Create COCO object
        json_path_dict = {
            'train': 'annotations/instances_train2014.json',
            'val': 'annotations/instances_val2014.json',
            'minival': 'annotations/instances_minival2014.json',
            'val35k': 'annotations/instances_valminusminival2014.json',
        }
        coco = COCO(os.path.join(dataset_dir, json_path_dict[subset]))

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        self.add_class('objects', 1, 'object')

        # Add images
        for i in image_ids:
            self.add_image(
                'objects',
                image_id=i,
                path=os.path.join(
                    image_dir,
                    coco.imgs[i]['file_name']),
                width=coco.imgs[i]['width'],
                height=coco.imgs[i]['height'],
                annotations=coco.loadAnns(
                    coco.getAnnIds(
                        imgIds=[i],
                        iscrowd=False)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]['annotations']
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id('objects.1')
            if class_id:
                m = self.annToMask(annotation, image_info['height'],
                                   image_info['width'])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                instance_masks.append(m)
                class_ids.append(self.class_names.index('object'))

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__).load_mask(image_id)
