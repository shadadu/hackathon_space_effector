#!/usr/bin/env python3
import math
import threading
import time
import rospy
import tf2_ros

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from shape_msgs.msg import SolidPrimitive


# ------------------------------
# Utility helpers
# ------------------------------

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


# ------------------------------
# Intercept Planner
# ------------------------------

class InterceptPlanner:

    def __init__(self):

        # Parameters
        self.object_topic = rospy.get_param("~object_topic", "/object/state")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.group_name = rospy.get_param("~group_name", "panda_arm")

        self.plan_service = rospy.get_param("~plan_service", "/plan_kinematic_path")
        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")

        self.latency_s = rospy.get_param("~latency_s", 0.15)
        self.t_min = rospy.get_param("~t_min", 0.3)
        self.t_max = rospy.get_param("~t_max", 3.0)
        self.t_step = rospy.get_param("~t_step", 0.05)

        self.v_ee_max = rospy.get_param("~v_ee_max", 0.6)

        self.pos_tol = rospy.get_param("~pos_tol", 1.0)
        self.ang_tol = rospy.get_param("~ang_tol", 3.142)

        self.allowed_planning_time = rospy.get_param("~allowed_planning_time", 6.0)

        self.start_state = panda_extended_open_start_state()

        self.last_odom = None
        self.busy = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        rospy.wait_for_service(self.ik_service)
        rospy.wait_for_service(self.plan_service)

        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)
        self.plan = rospy.ServiceProxy(self.plan_service, GetMotionPlan)

        rospy.Subscriber(self.object_topic, Odometry, self.cb_object)

        self.pub_goal = rospy.Publisher("/intercept/goal_pose", PoseStamped, queue_size=1)
        self.pub_pred = rospy.Publisher("/intercept/object_pred", PoseStamped, queue_size=1)
        rospy.loginfo("Goal pose and Object pred %s --> %s", self.pub_goal, self.pub_pred)

        # Start robust control loop thread
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

        rospy.loginfo("Robust InterceptPlanner running.")

    # ------------------------------

    def cb_object(self, msg):
        self.last_odom = msg

    # ------------------------------

    def loop(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():
            if not self.busy:
                self.process_once()
            rate.sleep()

    # ------------------------------
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
            except Exception as e:
                rospy.logerr("Planner exception: %s", str(e))

        th = threading.Thread(target=worker)
        th.start()
        th.join(timeout_s)

        if th.is_alive():
            rospy.logwarn("Planner timeout.")
            return None

        return result["resp"]

    def process_once(self):

        if self.last_odom is None:
            return

        choice = self.choose_intercept_time(self.last_odom)

        if choice is None:
            return

        t_hit, goal = choice

        rospy.loginfo("Intercept @ %.2fs with goal position %s and orientation %s",
                      t_hit, goal.pose.position, goal.pose.orientation)

        self.busy = True
        try:
            resp = self.call_planner_with_timeout(goal, self.allowed_planning_time + 1.0)
            if resp:
                rospy.loginfo("Planner returned: %d", resp.error_code.val)
        finally:
            self.busy = False

    # ------------------------------

    def effective_dt(self, odom, t_hit):
        now = rospy.Time.now()
        msg_t = odom.header.stamp if odom.header.stamp != rospy.Time(0) else now
        age = (now - msg_t).to_sec()
        return max(0.0, age + self.latency_s + t_hit)

    # ------------------------------

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

    # ------------------------------

    def choose_intercept_time(self, odom):

        ee0 = self.ee_position()
        if ee0 is None:
            return None

        t = self.t_min
        best = None

        while t <= self.t_max:

            obj_pred = self.predict_object_pose(odom, t)
            goal = obj_pred

            dx = goal.pose.position.x - ee0[0]
            dy = goal.pose.position.y - ee0[1]
            dz = goal.pose.position.z - ee0[2]

            dist = math.sqrt(dx * dx + dy * dy + dz * dz)

            if dist <= self.v_ee_max * t:
                if self.ik_feasible(goal):
                    best = (t, goal)
                    break

            t += self.t_step

        return best

    # ------------------------------

    def ee_position(self):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.ee_link,
                rospy.Time(0), rospy.Duration(0.2)
            )
            t = tf.transform.translation
            return t.x, t.y, t.z
        except:
            return None

    # ------------------------------

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
    rospy.init_node("intercept_planner")
    InterceptPlanner()
    rospy.spin()


if __name__ == "__main__":
    main()
