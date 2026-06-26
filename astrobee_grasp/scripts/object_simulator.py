#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


class ObjectSimulator:
    def __init__(self):
        # Topics / frames
        self.topic = rospy.get_param("~topic", "/object/state")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.object_frame = rospy.get_param("~object_frame", "object_link")

        # Publish rate
        self.rate_hz = float(rospy.get_param("~rate_hz", 20.0))

        # Motion mode. For HJB/DGM constant-drift tests, use drift=True and bounce=False.
        self.drift = as_bool(rospy.get_param("~drift", False))
        self.bounce = as_bool(rospy.get_param("~bounce", False))
        self.clamp_to_bounds = as_bool(rospy.get_param("~clamp_to_bounds", False))

        # Safe default pose for Panda reach testing
        self.x = float(rospy.get_param("~init_x", 0.50))
        self.y = float(rospy.get_param("~init_y", 0.00))
        self.z = float(rospy.get_param("~init_z", 0.25))

        # Constant velocity used only when drift=True.
        self.vx = float(rospy.get_param("~vx", -0.02))
        self.vy = float(rospy.get_param("~vy", 0.01))
        self.vz = float(rospy.get_param("~vz", 0.00))

        # Workspace box to keep object in reachable region
        self.x_min = float(rospy.get_param("~x_min", 0.35))
        self.x_max = float(rospy.get_param("~x_max", 0.65))
        self.y_min = float(rospy.get_param("~y_min", -0.20))
        self.y_max = float(rospy.get_param("~y_max", 0.20))
        self.z_min = float(rospy.get_param("~z_min", 0.10))
        self.z_max = float(rospy.get_param("~z_max", 0.40))

        # Fixed orientation by default
        self.roll = float(rospy.get_param("~roll", 0.0))
        self.pitch = float(rospy.get_param("~pitch", 0.0))
        self.yaw = float(rospy.get_param("~yaw", 0.0))

        # Covariances
        self.pos_cov = float(rospy.get_param("~pos_cov", 1e-3))
        self.rot_cov = float(rospy.get_param("~rot_cov", 1e-2))
        self.twist_cov = float(rospy.get_param("~twist_cov", 1e-3))

        self.pub = rospy.Publisher(self.topic, Odometry, queue_size=10)
        self.last_t = rospy.Time.now()

        rospy.loginfo(
            "object_simulator started: topic=%s drift=%s bounce=%s clamp=%s init=(%.3f, %.3f, %.3f) vel=(%.3f, %.3f, %.3f)",
            self.topic, str(self.drift), str(self.bounce), str(self.clamp_to_bounds),
            self.x, self.y, self.z, self.vx, self.vy, self.vz
        )

    def step(self, dt):
        if not self.drift:
            return

        self.x += self.vx * dt
        self.y += self.vy * dt
        self.z += self.vz * dt

        if self.bounce:
            if self.x < self.x_min:
                self.x = self.x_min
                self.vx *= -1.0
            elif self.x > self.x_max:
                self.x = self.x_max
                self.vx *= -1.0

            if self.y < self.y_min:
                self.y = self.y_min
                self.vy *= -1.0
            elif self.y > self.y_max:
                self.y = self.y_max
                self.vy *= -1.0

            if self.z < self.z_min:
                self.z = self.z_min
                self.vz *= -1.0
            elif self.z > self.z_max:
                self.z = self.z_max
                self.vz *= -1.0
        elif self.clamp_to_bounds:
            self.x = clamp(self.x, self.x_min, self.x_max)
            self.y = clamp(self.y, self.y_min, self.y_max)
            self.z = clamp(self.z, self.z_min, self.z_max)

    def make_msg(self):
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.object_frame

        qx, qy, qz, qw = quaternion_from_euler(self.roll, self.pitch, self.yaw)

        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = self.z
        odom.pose.pose.orientation = Quaternion(qx, qy, qz, qw)

        odom.twist.twist.linear.x = self.vx if self.drift else 0.0
        odom.twist.twist.linear.y = self.vy if self.drift else 0.0
        odom.twist.twist.linear.z = self.vz if self.drift else 0.0

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        pose_cov = [0.0] * 36
        twist_cov = [0.0] * 36
        pose_cov[0] = self.pos_cov
        pose_cov[7] = self.pos_cov
        pose_cov[14] = self.pos_cov
        pose_cov[21] = self.rot_cov
        pose_cov[28] = self.rot_cov
        pose_cov[35] = self.rot_cov

        twist_cov[0] = self.twist_cov
        twist_cov[7] = self.twist_cov
        twist_cov[14] = self.twist_cov
        twist_cov[21] = self.twist_cov
        twist_cov[28] = self.twist_cov
        twist_cov[35] = self.twist_cov

        odom.pose.covariance = pose_cov
        odom.twist.covariance = twist_cov
        return odom

    def run(self):
        rate = rospy.Rate(self.rate_hz)
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            dt = (now - self.last_t).to_sec()
            self.last_t = now
            if dt < 0.0:
                dt = 0.0

            self.step(dt)
            self.pub.publish(self.make_msg())
            rate.sleep()


def main():
    rospy.init_node("object_simulator")
    ObjectSimulator().run()


if __name__ == "__main__":
    main()
