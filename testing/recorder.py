import os, sys
import numpy as np
import cv2
import rospy
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

i = 0

def callback(img_data, depth_data):
    global i
    try:
        depth = bridge.imgmsg_to_cv2(depth_data)
        image = bridge.imgmsg_to_cv2(img_data, "bgr8")
        cv2.imwrite(os.path.join(sys.argv[-1], 'image', str(i).zfill(5)) + '.png', image)
        cv2.imwrite(os.path.join(sys.argv[-1], 'depth', str(i).zfill(5)) + '.png', depth)
        i += 1
    except CvBridgeError as e:
        print(e)

bridge = CvBridge()
image_sub = message_filters.Subscriber("/camera/rgb/image_raw", Image, queue_size=1)
depth_sub = message_filters.Subscriber("/camera/depth/image_raw", Image, queue_size=1)
ts = message_filters.ApproximateTimeSynchronizer([image_sub, depth_sub], 1, 1 / 300.0)
ts.registerCallback(callback)
pub = rospy.Publisher('/camera/instances/image_raw', Image, queue_size=10)

if __name__ == '__main__':
    rospy.init_node('rgbd_recorder', anonymous=True)
    rospy.spin()
