import os
import sys
import time

import cv2
import message_filters
import numpy as np
import rospy
import tensorflow as tf
import thread
from cv_bridge import CvBridge, CvBridgeError
from misc import draw_masks
from sensor_msgs.msg import Image

parentPath = os.path.abspath('..')
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import model as modellib
from instance_segmentation.object_config import Config


# Root directory of the project
ROOT_DIR = parentPath
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, 'logs')
# WEIGHTS_PATH = os.path.join(MODEL_DIR, "mask_rcnn_seg_scenenet_0259.h5")
WEIGHTS_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_seg_scenenn_0236.h5')


class InferenceConfig(Config):
    MODE = 'RGBD'
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 1220.7])


inference_config = InferenceConfig()
# Recreate the model in inference mode
model = modellib.MaskRCNN(
    mode='inference',
    config=inference_config,
    model_dir=MODEL_DIR)
model.load_weights(WEIGHTS_PATH, by_name=True)

graph = tf.get_default_graph()


class Shared:
    img_data = []
    depth_data = []
    running = False
    image = None
    rgbd = None


shared = Shared()


def main(shared):
    shared.running = True
    image = shared.image
    rgbd = shared.rgbd
    masks = []

    with graph.as_default():
        try:
            start = time.clock()
            result = model.detect([rgbd])[0]
            print(time.clock() - start)
            shared.running = False
        except BaseException:
            print('error in detection')
            shared.running = False
            return
    masks = result['masks']
    scores = result['scores']
    boxes = result['rois']
    img, vertices = draw_masks(image, masks)
    for i, (verts, color) in enumerate(vertices):
        cv2.polylines(img, np.int32([verts]), True, color * 255)
    for i, score in enumerate(scores):
        y1, x1, y2, x2 = boxes[i]
        cv2.putText(img, str(score), (x1, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))


def callback(img_data, depth_data):
    if not shared.running:
        thread.start_new_thread(main, (shared,))
    try:
        depth = bridge.imgmsg_to_cv2(depth_data)
        shared.image = bridge.imgmsg_to_cv2(img_data, 'bgr8')
        shared.rgbd = np.dstack((shared.image, depth))
    except CvBridgeError as e:
        print(e)


bridge = CvBridge()
image_sub = message_filters.Subscriber(
    '/camera/rgb/image_raw', Image, queue_size=1)
depth_sub = message_filters.Subscriber(
    '/camera/depth/image_raw', Image, queue_size=1)
ts = message_filters.ApproximateTimeSynchronizer(
    [image_sub, depth_sub], 1, 1 / 300.0)
ts.registerCallback(callback)
pub = rospy.Publisher('/camera/instances/image_raw', Image, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('mask_rcnn', anonymous=True)
    rospy.spin()
