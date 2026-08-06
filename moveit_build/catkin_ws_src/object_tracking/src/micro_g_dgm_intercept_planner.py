#!/usr/bin/env python3
import math
import threading

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import Constraints, MotionPlanRequest, OrientationConstraint, PositionConstraint, RobotState
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest, GetPositionIK, GetPositionIKRequest
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive


def make_robot_state_from_joint_dict(joint_dict):
    js = JointState()
    js.name = list(joint_dict.keys())
    js.position = [joint_dict[n] for n in js.name]
    rs = RobotState()
    rs.joint_state = js
    return rs


def panda_extended_open_start_state():
    joints = {
        "panda_joint1": 0.0,
        "panda_joint2": 0.0,
        "panda_joint3": 0.0,
        "panda_joint4": 0.0,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        "panda_finger_joint1": 0.035,
        "panda_finger_joint2": 0.035,
    }
    return make_robot_state_from_joint_dict(joints)


def make_goal_constraints(goal_pose, ee_link, pos_tol, ang_tol):
    c = Constraints()

    pc = PositionConstraint()
    pc.header = goal_pose.header
    pc.link_name = ee_link
    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [pos_tol, pos_tol, pos_tol]
    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal_pose.pose)
    c.position_constraints.append(pc)

    oc = OrientationConstraint()
    oc.header = goal_pose.header
    oc.link_name = ee_link
    oc.orientation = goal_pose.pose.orientation
    oc.absolute_x_axis_tolerance = ang_tol
    oc.absolute_y_axis_tolerance = ang_tol
    oc.absolute_z_axis_tolerance = ang_tol
    c.orientation_constraints.append(oc)

    return c


class MicroGDGMInterceptPlanner:
    def __init__(self):
        self.object_topic = rospy.get_param("~object_topic", "/object/state")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.plan_service = rospy.get_param("~plan_service", "/micro_g_dgm/get_motion_plan")
        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")
        self.service_wait_timeout = float(rospy.get_param("~service_wait_timeout", 20.0))

        self.latency_s = float(rospy.get_param("~latency_s", 0.15))
        self.object_max_age_s = float(rospy.get_param("~object_max_age_s", 0.50))
        self.t_min = float(rospy.get_param("~t_min", 0.3))
        self.t_max = float(rospy.get_param("~t_max", 3.0))
        self.t_step = float(rospy.get_param("~t_step", 0.05))
        self.v_ee_max = float(rospy.get_param("~v_ee_max", 0.6))
        self.v_base_max = float(rospy.get_param("~v_base_max", 0.08))
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.20))
        self.ang_tol = float(rospy.get_param("~ang_tol", 3.142))
        self.allowed_planning_time = float(rospy.get_param("~allowed_planning_time", 6.0))

        self.start_state = panda_extended_open_start_state()
        self.last_odom = None
        self.busy = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.wait_for_required_service(self.ik_service, "MoveIt IK")
        self.wait_for_required_service(
            self.plan_service,
            "micro-g DGM planner (start micro_g_dgm_planner_node.py first)",
        )
        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)
        self.plan = rospy.ServiceProxy(self.plan_service, GetMotionPlan)

        rospy.Subscriber(self.object_topic, Odometry, self.cb_object, queue_size=1)
        self.pub_goal = rospy.Publisher("/intercept/goal_pose", PoseStamped, queue_size=1)
        self.pub_pred = rospy.Publisher("/intercept/object_pred", PoseStamped, queue_size=1)

        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        rospy.loginfo("Micro-g DGM InterceptPlanner running.")

    def wait_for_required_service(self, service_name, description):
        rospy.loginfo(
            "Waiting up to %.1fs for %s service: %s",
            self.service_wait_timeout,
            description,
            service_name,
        )
        try:
            rospy.wait_for_service(service_name, timeout=self.service_wait_timeout)
        except rospy.ROSException as exc:
            raise rospy.ROSInitException(
                "Timed out waiting for {} at '{}': {}".format(
                    description, service_name, exc
                )
            )
        rospy.loginfo("Connected to service: %s", service_name)

    def cb_object(self, msg):
        self.last_odom = msg

    def object_fresh(self, odom):
        stamp = odom.header.stamp if odom.header.stamp != rospy.Time(0) else rospy.Time.now()
        age = (rospy.Time.now() - stamp).to_sec()
        return 0.0 <= age <= self.object_max_age_s

    def loop(self):
        max_attempts = 4
        attempts = 0
        rate = rospy.Rate(float(rospy.get_param("~loop_hz", 2.0)))
        while not rospy.is_shutdown() and attempts <= max_attempts:
            if not self.busy:
                resp = self.process_once()
                rospy.logdebug("process_once returned: %s", resp)
            attempts += 1
            rate.sleep()

    def process_once(self):
        odom = self.last_odom
        if odom is None or not self.object_fresh(odom):
            return

        choice = self.choose_intercept_time(odom)
        if choice is None:
            return

        t_hit, goal = choice
        self.pub_pred.publish(goal)
        self.pub_goal.publish(goal)
        rospy.loginfo(
            "Micro-g intercept @ %.2fs goal=(%.3f, %.3f, %.3f)",
            t_hit, goal.pose.position.x, goal.pose.position.y, goal.pose.position.z,
        )

        self.busy = True
        try:
            resp = self.call_planner_with_timeout(goal, self.allowed_planning_time + 1.0)
            if resp:
                rospy.loginfo("Planner returned: %d", resp.error_code.val)
        finally:
            self.busy = False

    def call_planner_with_timeout(self, goal, timeout_s):
        result = {"resp": None}

        def worker():
            try:
                mpr = MotionPlanRequest()
                mpr.group_name = self.group_name
                mpr.allowed_planning_time = self.allowed_planning_time
                mpr.start_state = self.start_state
                mpr.goal_constraints = [
                    make_goal_constraints(goal, self.ee_link, self.pos_tol, self.ang_tol)
                ]
                req = GetMotionPlanRequest()
                req.motion_plan_request = mpr
                result["resp"] = self.plan(req).motion_plan_response
            except Exception as exc:
                rospy.logerr("Planner exception: %s", exc)

        th = threading.Thread(target=worker)
        th.start()
        th.join(timeout_s)
        if th.is_alive():
            rospy.logwarn("Planner timeout.")
            return None
        return result["resp"]

    def effective_dt(self, odom, t_hit):
        now = rospy.Time.now()
        msg_t = odom.header.stamp if odom.header.stamp != rospy.Time(0) else now
        age = (now - msg_t).to_sec()
        return max(0.0, age + self.latency_s + t_hit)

    def predict_object_pose(self, odom, t_hit):
        dt = self.effective_dt(odom, t_hit)
        p0 = odom.pose.pose.position
        v = odom.twist.twist.linear

        ps = PoseStamped()
        ps.header.frame_id = odom.header.frame_id or self.world_frame
        ps.header.stamp = rospy.Time.now()
        ps.pose.position.x = p0.x + v.x * dt
        ps.pose.position.y = p0.y + v.y * dt
        ps.pose.position.z = p0.z + v.z * dt
        ps.pose.orientation = odom.pose.pose.orientation
        return ps

    def choose_intercept_time(self, odom):
        ee0 = self.ee_position()
        if ee0 is None:
            return None

        t = self.t_min
        best = None
        max_closing_speed = self.v_ee_max + self.v_base_max

        while t <= self.t_max:
            goal = self.predict_object_pose(odom, t)
            dx = goal.pose.position.x - ee0[0]
            dy = goal.pose.position.y - ee0[1]
            dz = goal.pose.position.z - ee0[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist <= max_closing_speed * t and self.ik_feasible(goal):
                best = (t, goal)
                break
            t += self.t_step

        return best

    def ee_position(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.ee_link, rospy.Time(0), rospy.Duration(0.2)
            )
            t = tf.transform.translation
            return t.x, t.y, t.z
        except Exception:
            return None

    def ik_feasible(self, goal):
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped = goal
        req.ik_request.robot_state = self.start_state
        req.ik_request.timeout = rospy.Duration(2)
        resp = self.ik(req)
        return resp.error_code.val == 1


def main():
    rospy.init_node("micro_g_dgm_intercept_planner")
    MicroGDGMInterceptPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()
