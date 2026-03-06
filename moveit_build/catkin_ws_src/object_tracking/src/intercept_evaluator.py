#!/usr/bin/env python3
import math
import rospy
import tf2_ros

from geometry_msgs.msg import TransformStamped, Vector3
from object_tracking.msg import InterceptMetrics


def quat_conj(q):
    return (-q[0], -q[1], -q[2], q[3])


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def quat_angle(q):
    # assumes q normalized; returns rotation angle in [0,pi]
    w = max(-1.0, min(1.0, q[3]))
    return 2.0 * math.acos(abs(w))


class InterceptEvaluator:
    def __init__(self):
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.ee_frame = rospy.get_param("~ee_frame", "panda_hand")
        self.object_frame = rospy.get_param("~object_frame", "object_link")
        self.rate_hz = float(rospy.get_param("~rate_hz", 50.0))
        self.tf_timeout = rospy.Duration(rospy.get_param("~tf_timeout_s", 0.05))

        self.tfbuf = tf2_ros.Buffer(cache_time=rospy.Duration(3.0))
        self.tfl = tf2_ros.TransformListener(self.tfbuf)

        self.pub = rospy.Publisher("/intercept/eval/metrics", InterceptMetrics, queue_size=10)

        self.timer = rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self.on_timer)
        rospy.loginfo("InterceptEvaluator up. world=%s ee=%s object=%s",
                      self.world_frame, self.ee_frame, self.object_frame)

    def lookup(self, target, source):
        # target <- source
        return self.tfbuf.lookup_transform(target, source, rospy.Time(0), self.tf_timeout)

    def on_timer(self, _evt):
        try:
            Tw_e = self.lookup(self.world_frame, self.ee_frame)  # world<-ee
            Tw_o = self.lookup(self.world_frame, self.object_frame)  # world<-object
            rospy.loginfo("Proximity metrics: world->end-effector=%s ; world->object=%s", str(Tw_e), str(Tw_o))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF lookup failed: %s", str(e))
            return

        ex = Tw_e.transform.translation.x
        ey = Tw_e.transform.translation.y
        ez = Tw_e.transform.translation.z

        ox = Tw_o.transform.translation.x
        oy = Tw_o.transform.translation.y
        oz = Tw_o.transform.translation.z

        dx = ox - ex
        dy = oy - ey
        dz = oz - ez
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        rospy.loginfo("Proximity metrics: euclidean dist=%s", dist)

        # orientation difference (optional)
        qe = (
            Tw_e.transform.rotation.x,
            Tw_e.transform.rotation.y,
            Tw_e.transform.rotation.z,
            Tw_e.transform.rotation.w,
        )
        qo = (
            Tw_o.transform.rotation.x,
            Tw_o.transform.rotation.y,
            Tw_o.transform.rotation.z,
            Tw_o.transform.rotation.w,
        )
        qerr = quat_mul(quat_conj(qe), qo)
        ang = quat_angle(qerr)

        rospy.loginfo("Proximity metrics: euclidean dist=%s orientation diff=%s", dist, ang)

        msg = InterceptMetrics()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.world_frame
        msg.world_frame = self.world_frame
        msg.ee_frame = self.ee_frame
        msg.object_frame = self.object_frame
        msg.distance_m = float(dist)
        msg.angle_rad = float(ang)
        msg.rel_pos = Vector3(dx, dy, dz)
        self.pub.publish(msg)


def main():
    rospy.init_node("intercept_evaluator")
    InterceptEvaluator()
    rospy.spin()


if __name__ == "__main__":
    main()
