#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from filterpy.kalman import KalmanFilter

class ObjectEKF:
    def __init__(self):
        self.dt = rospy.get_param("ekf/dt", 0.02)

        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = np.zeros(6)

        self.kf.F = np.block([
            [np.eye(3), self.dt * np.eye(3)],
            [np.zeros((3,3)), np.eye(3)]
        ])

        q_pos = rospy.get_param("ekf/process_noise/position", 1e-6)
        q_vel = rospy.get_param("ekf/process_noise/velocity", 1e-4)
        self.kf.Q = np.diag([q_pos]*3 + [q_vel]*3)

        r_pos = rospy.get_param("ekf/measurement_noise/position", 5e-4)
        self.kf.R = np.diag([r_pos]*3)

        self.kf.H = np.block([np.eye(3), np.zeros((3,3))])

        self.sub = rospy.Subscriber(
            "/object/pose_tracked",
            PoseWithCovarianceStamped,
            self.cb
        )

        self.pub = rospy.Publisher(
            "/object/state",
            Odometry,
            queue_size=1
        )

        self.last_time = rospy.Time.now()

    def cb(self, msg):
        z = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

        now = msg.header.stamp
        self.dt = (now - self.last_time).to_sec()
        self.last_time = now

        self.kf.predict()
        self.kf.update(z)

        self.publish_state(msg.header.stamp)

    def publish_state(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"

        odom.pose.pose.position.x = self.kf.x[0]
        odom.pose.pose.position.y = self.kf.x[1]
        odom.pose.pose.position.z = self.kf.x[2]

        self.pub.publish(odom)

if __name__ == "__main__":
    rospy.init_node("object_ekf")
    ObjectEKF()
    rospy.spin()
