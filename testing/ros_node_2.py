import os, sys
import time
import thread
import numpy as np
import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf

parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

import model as modellib
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
    img_data = []
    depth_data = []
    running = False

shared = Shared()

def main(shared):
    shared.running = True
    image = []
    depth = []
    masks = []
    try:
        image = bridge.imgmsg_to_cv2(shared.img_data, "bgr8")
        depth = bridge.imgmsg_to_cv2(shared.depth_data)
    except CvBridgeError as e:
        print(e)

    rgbd = np.dstack((image, depth))
    with graph.as_default():
        try:
            start = time.clock()
            masks = model.detect([rgbd])[0]['masks']
            print(time.clock() - start)
            shared.running = False
        except:
            print('error in detection')
            shared.running = False
            return
    img, vertices = draw_masks(image, masks)
    for verts, color in vertices:
        cv2.polylines(img, np.int32([verts]), True, color * 255)
    pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))

def callback(img_data, depth_data):
    if shared.running:
        return
    shared.img_data = img_data
    shared.depth_data = depth_data
    thread.start_new_thread(main, (shared,))
    

bridge = CvBridge()
image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image, queue_size=1)
depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image, queue_size=1)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 1 / 300.0)
ts.registerCallback(callback)
pub = rospy.Publisher('/camera/instances/image_raw', Image, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('mask_rcnn', anonymous=True)
    rospy.spin()
