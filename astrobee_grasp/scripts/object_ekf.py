#!/usr/bin/env python3
"""
object_ekf.py

Microgravity-friendly object state estimator.

Inputs:
  /object/pose_tracked (geometry_msgs/PoseWithCovarianceStamped)

Outputs:
  /object/state (nav_msgs/Odometry)

State:
  x = [px, py, pz, vx, vy, vz]^T  (constant velocity)
Orientation:
  Taken from latest measurement (optionally smoothed), not EKF'd here.
"""

import rospy
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class ObjectEKFNode:
    def __init__(self):
        # -------- Parameters --------
        self.dt = float(rospy.get_param("ekf/dt", 0.02))
        self.pub_rate = float(rospy.get_param("ekf/pub_rate", 10.0))

        self.world_frame = rospy.get_param("ekf/frames/world", "world")
        self.object_frame = rospy.get_param("ekf/frames/object", "object_link")

        # Noise (tunable)
        q_pos = float(rospy.get_param("ekf/process_noise/position", 1e-6))
        q_vel = float(rospy.get_param("ekf/process_noise/velocity", 1e-4))
        r_pos = float(rospy.get_param("ekf/measurement_noise/position", 5e-4))

        # Initial covariance
        p0_pos = float(rospy.get_param("ekf/initial_covariance/position", 1e-3))
        p0_vel = float(rospy.get_param("ekf/initial_covariance/velocity", 1e-2))

        # Jump gating
        self.max_position_jump = float(rospy.get_param("ekf/max_position_jump", 0.25))

        # Orientation smoothing (0 = no smoothing, 1 = never update)
        self.ori_smoothing = float(rospy.get_param("ekf/orientation_smoothing", 0.2))
        self.ori_smoothing = clamp(self.ori_smoothing, 0.0, 0.99)

        # -------- Filter Matrices --------
        # State: [p; v]
        self.x = np.zeros((6, 1), dtype=np.float64)

        self.P = np.diag([p0_pos, p0_pos, p0_pos, p0_vel, p0_vel, p0_vel]).astype(np.float64)

        # Measurement z = position only
        self.H = np.hstack([np.eye(3), np.zeros((3, 3))]).astype(np.float64)
        self.R = np.diag([r_pos, r_pos, r_pos]).astype(np.float64)

        # Process noise
        self.Q_base = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float64)

        # Timing
        self.last_meas_time = None
        self.have_meas = False

        # Latest orientation (quaternion) from measurement
        self.q = (0.0, 0.0, 0.0, 1.0)

        # -------- ROS I/O --------
        self.sub = rospy.Subscriber(
            "/object/pose_tracked",
            PoseWithCovarianceStamped,
            self.meas_cb,
            queue_size=1
        )

        self.pub = rospy.Publisher("/object/state", Odometry, queue_size=1)

        # Publish at fixed rate (important for MoveIt integration)
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.pub_rate), self.on_timer)

        rospy.loginfo("object_ekf: started. Publishing /object/state at %.2f Hz", self.pub_rate)

    # ----------------- Core KF -----------------
    def predict(self, dt):
        # F for constant velocity
        F = np.block([
            [np.eye(3), dt * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)]
        ]).astype(np.float64)

        # Scale Q with dt (simple heuristic)
        Q = self.Q_base.copy()
        Q[0:3, 0:3] *= max(dt, 1e-6)
        Q[3:6, 3:6] *= max(dt, 1e-6)

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_pos(self, z):
        # z is (3,1)
        y = z - (self.H @ self.x)  # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P

    # ----------------- Orientation Handling -----------------
    def smooth_quat(self, q_new):
        # Simple linear blend + renormalize (not true slerp, but stable)
        ax = self.ori_smoothing
        q_old = np.array(self.q, dtype=np.float64)
        q_new = np.array(q_new, dtype=np.float64)

        q_blend = (ax * q_old) + ((1.0 - ax) * q_new)
        n = np.linalg.norm(q_blend)
        if n < 1e-12:
            return self.q
        q_blend /= n
        return (float(q_blend[0]), float(q_blend[1]), float(q_blend[2]), float(q_blend[3]))

    # ----------------- ROS Callbacks -----------------
    def meas_cb(self, msg: PoseWithCovarianceStamped):
        # Frame warning (don’t hard-fail; just warn)
        if msg.header.frame_id and msg.header.frame_id != self.world_frame:
            rospy.logwarn_throttle(
                2.0,
                "object_ekf: pose_tracked frame_id='%s' but expected '%s'. Continuing.",
                msg.header.frame_id, self.world_frame
            )

        t = msg.header.stamp if msg.header.stamp != rospy.Time(0) else rospy.Time.now()

        # dt from measurement timestamps when possible
        if self.last_meas_time is None:
            dt = self.dt
        else:
            dt = (t - self.last_meas_time).to_sec()
            if dt <= 0.0 or dt > 1.0:
                # protect against clock jumps / pauses
                dt = self.dt

        self.last_meas_time = t

        # Extract measured position
        p = msg.pose.pose.position
        z = np.array([[p.x], [p.y], [p.z]], dtype=np.float64)

        # Jump rejection (if we already have a measurement baseline)
        if self.have_meas:
            pred_pos = (self.H @ self.x)
            jump = np.linalg.norm(z - pred_pos)
            if jump > self.max_position_jump:
                rospy.logwarn_throttle(
                    1.0,
                    "object_ekf: rejecting position jump %.3f m > %.3f m",
                    float(jump), self.max_position_jump
                )
                # Still predict forward so state doesn't freeze
                self.predict(dt)
                # Update orientation anyway (optional)
                qn = msg.pose.pose.orientation
                self.q = self.smooth_quat((qn.x, qn.y, qn.z, qn.w))
                return

        # Normal predict/update
        self.predict(dt)
        self.update_pos(z)

        # Update orientation
        qn = msg.pose.pose.orientation
        self.q = self.smooth_quat((qn.x, qn.y, qn.z, qn.w))

        self.have_meas = True

        rospy.loginfo_throttle(2.0, "object_ekf: received pose_tracked @ ~%.2f Hz", 1.0 / max(dt, 1e-6))

    def on_timer(self, _evt):
        # If no measurements yet, still publish a predictable state (zeros)
        # If measurements exist but stop arriving, keep predicting forward with nominal dt
        if self.have_meas:
            self.predict(1.0 / self.pub_rate)

        self.publish_state(rospy.Time.now())

    def publish_state(self, stamp):
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.object_frame

        # Pose
        odom.pose.pose.position.x = float(self.x[0, 0])
        odom.pose.pose.position.y = float(self.x[1, 0])
        odom.pose.pose.position.z = float(self.x[2, 0])

        odom.pose.pose.orientation.x = self.q[0]
        odom.pose.pose.orientation.y = self.q[1]
        odom.pose.pose.orientation.z = self.q[2]
        odom.pose.pose.orientation.w = self.q[3]

        # Twist
        odom.twist.twist.linear.x = float(self.x[3, 0])
        odom.twist.twist.linear.y = float(self.x[4, 0])
        odom.twist.twist.linear.z = float(self.x[5, 0])

        # Covariance (fill position/velocity blocks conservatively)
        cov = np.zeros((6, 6), dtype=np.float64)
        cov[0:3, 0:3] = self.P[0:3, 0:3]
        cov[3:6, 3:6] = self.P[3:6, 3:6]
        odom.pose.covariance = cov.reshape(-1).tolist()

        self.pub.publish(odom)


if __name__ == "__main__":
    rospy.init_node("object_ekf")
    ObjectEKFNode()
    rospy.spin()
