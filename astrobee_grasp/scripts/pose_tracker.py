#!/usr/bin/env python3

import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as R

class PoseTracker:
    def __init__(self):
        self.last_pose = None
        self.max_translation_jump = 0.2      # meters
        self.max_rotation_jump = np.deg2rad(20)

        rospy.Subscriber("/object/pose_raw",
                         PoseWithCovarianceStamped,
                         self.cb)

        self.pub = rospy.Publisher("/object/pose_tracked",
                                   PoseWithCovarianceStamped,
                                   queue_size=1)

    def cb(self, msg):
        pose = msg.pose.pose

        p = np.array([pose.position.x,
                      pose.position.y,
                      pose.position.z])

        q = np.array([pose.orientation.x,
                      pose.orientation.y,
                      pose.orientation.z,
                      pose.orientation.w])

        if self.last_pose is not None:
            p_last, q_last = self.last_pose

            dp = np.linalg.norm(p - p_last)

            rot_err = R.from_quat(q_last).inv() * R.from_quat(q)
            dtheta = rot_err.magnitude()

            if dp > self.max_translation_jump or dtheta > self.max_rotation_jump:
                rospy.logwarn("PoseTracker: Rejected outlier pose")
                return

        self.last_pose = (p, q)
        self.pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node("pose_tracker")
    PoseTracker()
    rospy.spin()
