#!/usr/bin/env python3
import json
import math
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler

def pixel_to_world(u, v, W, H, m_per_px, z0):
    # Origin at image center; +x right, +y up
    x = (u - 0.5 * W) * m_per_px
    y = (0.5 * H - v) * m_per_px
    z = z0
    return x, y, z

def make_odom(frame_id, child_frame_id, x, y, z, qxyzw, vx, vy, vz):
    msg = Odometry()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = frame_id
    msg.child_frame_id = child_frame_id

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z
    msg.pose.pose.orientation = Quaternion(*qxyzw)

    msg.twist.twist.linear.x = vx
    msg.twist.twist.linear.y = vy
    msg.twist.twist.linear.z = vz

    # Leave angular vel 0 for now
    msg.twist.twist.angular.x = 0.0
    msg.twist.twist.angular.y = 0.0
    msg.twist.twist.angular.z = 0.0
    return msg

def main():
    rospy.init_node("vlm_object_state_publisher")

    # Params (make these match your system)
    out_topic = rospy.get_param("~out_topic", "/object/state")
    frame_id = rospy.get_param("~frame_id", "world")
    child_frame_id = rospy.get_param("~child_frame_id", "object")

    W = float(rospy.get_param("~W", 640))
    H = float(rospy.get_param("~H", 480))
    m_per_px = float(rospy.get_param("~m_per_px", 1.0))
    z0 = float(rospy.get_param("~z0", 0.0))

    rate_hz = float(rospy.get_param("~rate", 30.0))

    # For demo: JSON passed as a ROS param (replace with your real VLM input)
    # Example:
    # rosparam set /vlm_object_state_publisher/json_string '{"center_of_mass_beginning":[500,500],...}'
    json_string = rospy.get_param("~json_string", "{}")

    pub = rospy.Publisher(out_topic, Odometry, queue_size=10)
    r = rospy.Rate(rate_hz)

    while not rospy.is_shutdown():
        try:
            data = json.loads(json_string)

            # Use "end" as current estimate (or choose beginning/end as needed)
            u, v = data.get("center_of_mass_end", [W * 0.5, H * 0.5])

            roll, pitch, yaw = data.get("eular_angle_orientation_end", [0.0, 0.0, 0.0])
            qx, qy, qz, qw = quaternion_from_euler(roll, pitch, yaw)

            vx, vy = data.get("velocity_horizontal_end", [0.0, 0.0])
            vz_pair = data.get("velocity_vertical_end", [0.0, 0.0])
            vz = float(vz_pair[0]) if isinstance(vz_pair, (list, tuple)) and len(vz_pair) > 0 else 0.0

            x, y, z = pixel_to_world(float(u), float(v), W, H, m_per_px, z0)
            msg = make_odom(frame_id, child_frame_id, x, y, z, (qx, qy, qz, qw), float(vx), float(vy), float(vz))
            pub.publish(msg)

        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Failed to publish VLM odom: {e}")

        r.sleep()

if __name__ == "__main__":
    main()