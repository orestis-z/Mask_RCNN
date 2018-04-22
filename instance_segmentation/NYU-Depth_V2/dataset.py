import os, sys
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../..")
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from instance_segmentation.objects_config import ObjectsConfig
from instance_segmentation.objects_dataset import ObjectsDataset, normalize
from data.names import names


class Config(ObjectsConfig):
    NAME = "NYU_Depth_V2_sceneNet"

    MODE = 'RGBD'
    BACKBONE = 'resnet50'
    # BACKBONE = 'resnet101'

    # IMAGE_MIN_DIM = 448
    # IMAGE_MAX_DIM = 448

    IMAGES_PER_GPU = 1
    LEARNING_RATE = 0.001
    # USE_MINI_MASK = False

    # NUM_CLASSES = 1 + 1

    # STEPS_PER_EPOCH = 10
    
    # Image mean (RGBD)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 255 / 2]) # 1220.7 / 1000, 255.0 / 100])


class Dataset(ObjectsDataset):
    EXCLUDE = ['floor', 'wall', 'ceiling'] # stuff
    # EXCLUDE = [] # stuff

    def load(self, dataset_dir, subset, skip=0):
        assert(subset == 'training' or subset == 'validation')

        dataset_dir = os.path.join(dataset_dir, subset)

        for idx, name in enumerate(names):
            self.add_class("NYU_Depth_V2_sceneNet", idx + 1, name)

        # self.add_class("NYU_Depth_V2_sceneNet", 1, 'object')

        count = 0
        exclude = set(['depths', 'instances', 'labels'])
        # Add images
        for i, (root, dirs, files) in enumerate(os.walk(dataset_dir, topdown=True)):
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
                        width, height = (640, 480)
                        self.add_image(
                            "NYU_Depth_V2_sceneNet",
                            image_id=i,
                            path=path,
                            depth_path=depth_path,
                            instances_path=instances_path,
                            labels_path=labels_path,
                            parent_path=parent_path,
                            width=width,
                            height=height)
                        count += 1
        print('added {} images for {}'.format(count, subset))

    def load_image(self, image_id, mode="RGBD", canny_args=(100, 200)):
        if self.use_generated:
            parent_path = self.image_info[image_id]['parent_path']
            file_name = self.image_info[image_id]['file_name']
            return np.load(os.path.join(parent_path, img_path, file_name + ".npy"))
        image = np.load(self.image_info[image_id]['path']).T
        if mode == "RGBDE":
            depth = np.load(self.image_info[image_id]['depth_path']).T
            depth = normalize(depth)
            edges = cv2.Canny(np.uint8(image), *canny_args)
            rgbde = np.dstack((image, depth, edges))
            return rgbde
        elif mode == "RGBD":
            depth = np.load(self.image_info[image_id]['depth_path']).T
            depth = normalize(depth)
            rgbd = np.dstack((image, depth))
            return rgbd
        else:
            return image

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
        # if 0 in instances:
        #     x = instances.index(0)
        #     del labels[x]
        #     del instances[x]
        for excl in self.EXCLUDE:
            idx = names.index(excl) + 1
            if idx in labels:
                x = labels.index(idx)
                del labels[x]
                del instances[x]
                print("rm {}".format(excl))
            # if idx in instances:
            #     x = instances.index(idx)
            #     del labels[x]
            #     del instances[x]

        unique_instances = np.stack((labels, instances))

        n_instances = unique_instances.shape[1]
        instances = np.repeat(np.expand_dims(instances_img, axis=2), n_instances, axis=2)
        labels = np.repeat(np.expand_dims(labels_img, axis=2), n_instances, axis=2)
        masks = self.to_mask_v(instances, labels, unique_instances[0], unique_instances[1])
        if not n_instances:
            raise ValueError("No instances for image {}".format(mask_path))

        class_ids = np.array(unique_instances[0], dtype=np.int32)
        # class_ids = np.array([1] * n_instances, dtype=np.int32)

        return masks, class_ids

if __name__ == '__main__':
    dataset = Dataset()
    dataset.load('/home/orestisz/repositories/Mask_RCNN/instance_segmentation/NYU-Depth_V2/data', 'validation')
    masks, class_ids = dataset.load_mask(0)

