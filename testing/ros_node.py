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
from visualize import Polygon
from instance_segmentation.object_config import Config
from misc import draw_masks

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
    with graph.as_default():
        try:
            start = time.clock()
            shared.masks = model.detect([rgbd])[0]['masks']
            print(time.clock() - start)
            shared.i_pred += 1
        except:
            print('error in detection')
    shared.running = False

def plot_loop(shared):
    ax = plt.subplots(1)[1]
    while True:
        if shared.i_pred > shared.i_plot:
            plt.cla()
            plt.axis("off")
            img, vertices = draw_masks(shared.image, shared.masks)
            for verts, color in vertices:
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
            plt.imshow(img)
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
