#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class TFBroadcaster:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/object/state", Odometry, self.cb)

    def cb(self, msg):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = "world"
        t.child_frame_id = "object_link"

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z

        # Optional: for orientation
        t.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node("object_tf")
    TFBroadcaster()
    rospy.spin()
