#!/usr/bin/env python3

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry

class TFBroadcaster:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()
        self.state_topic = rospy.get_param("~state_topic", "/object/state")
        self.parent_frame = rospy.get_param("~parent_frame", "world")
        self.child_frame = rospy.get_param("~child_frame", "object_link")

        rospy.Subscriber(self.state_topic, Odometry, self.cb, queue_size=10)

        rospy.loginfo("TFBroadcaster: subscribing to %s, publishing TF %s -> %s",
                      self.state_topic, self.parent_frame, self.child_frame)

    def cb(self, msg: Odometry):
        t = TransformStamped()

        # Prefer message stamp if available, else now
        t.header.stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time.now()

        # Parent frame: prefer msg.header.frame_id, else param default
        t.header.frame_id = msg.header.frame_id if msg.header.frame_id else self.parent_frame

        # Child frame fixed (param)
        t.child_frame_id = self.child_frame

        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation

        self.br.sendTransform(t)

if __name__ == "__main__":
    rospy.init_node("object_tf")
    TFBroadcaster()
    rospy.spin()