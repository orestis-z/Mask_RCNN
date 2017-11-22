import os, sys
import time
import random
import thread
import numpy as np
import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import matplotlib.pyplot as plt

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import model as modellib
from visualize import *
from instance_segmentation.object_config import Config

# Root directory of the project
ROOT_DIR = parentPath
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
WEIGHTS_DIR = os.path.join(MODEL_DIR, "mask_rcnn_seg_scenenn_0100.h5")

class InferenceConfig(Config):
    MODE = 'RGBD'
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])

inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", config=inference_config, model_dir=MODEL_DIR)
model.load_weights(WEIGHTS_DIR, by_name=True)

graph = tf.get_default_graph()

class Shared:
    masks = None
    image = None
    depth = None
    running = False
    i_pred = 0
    i_plot = 0

shared = Shared()

def main(shared):
    shared.running = True
    rgbd = np.dstack((shared.image, shared.depth))
    start = time.clock()
    with graph.as_default():
        try:
            shared.masks = model.detect([rgbd])[0]['masks']
            shared.i_pred += 1
        except:
            print('error in detection')
    print(time.clock() - start)
    shared.running = False

def display_masks(image, masks, ax):
    # Number of instances
    N = masks.shape[-1]

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    # plt.set_ylim(height + 10, -10)
    # plt.set_xlim(-10, width + 10)
    plt.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    plt.imshow(masked_image.astype(np.uint8))

def plot_loop(shared):
    ax = plt.subplots(1)[1]
    while True:
        if shared.i_pred > shared.i_plot:
            plt.cla()
            # plt.axis("off")
            display_masks(shared.image, shared.masks, ax)
            if shared.i_plot == 0:
                plt.show(block=False)
            else:
                plt.draw()
                plt.pause(0.01)
            shared.i_plot += 1

class image_converter:
    def __init__(self):
        self.bridge = CvBridge()
        image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image)
        depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 10, 1 / 300.0)
        self.ts.registerCallback(self.callback)

    def callback(self, img_data, depth_data):
        if not shared.running and shared.i_pred == shared.i_plot:
            try:
                shared.image = self.bridge.imgmsg_to_cv2(img_data, "bgr8")
                shared.depth = self.bridge.imgmsg_to_cv2(depth_data)
            except CvBridgeError as e:
                print(e)
                
            thread.start_new_thread(main, (shared,))

def listener():
    ic = image_converter()
    rospy.init_node('mask_rcnn', anonymous=True)
    rospy.spin()

if __name__ == '__main__':
    thread.start_new_thread(plot_loop, (shared,))
    listener()
