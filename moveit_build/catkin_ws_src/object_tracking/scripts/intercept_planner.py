#!/usr/bin/env python3
import math
import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState

from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Vector3, Point, Quaternion
from std_msgs.msg import Header

def make_robot_state_from_joint_dict(joint_dict):
    js = JointState()
    js.name = list(joint_dict.keys())
    js.position = [joint_dict[n] for n in js.name]
    rs = RobotState()
    rs.joint_state = js
    return rs

def panda_extended_open_start_state():
    # From panda.srdf "extended" + "open"
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

def make_goal_constraints_from_pose(goal_pose_stamped: PoseStamped, ee_link: str,
                                   pos_tol=0.01, ang_tol=0.10):
    """
    Build MoveIt goal Constraints for an end-effector pose.
    """
    c = Constraints()

    # Position constraint (box)
    pc = PositionConstraint()
    pc.header = goal_pose_stamped.header
    pc.link_name = ee_link
    pc.target_point_offset = Vector3(0.0, 0.0, 0.0)

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [pos_tol, pos_tol, pos_tol]

    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal_pose_stamped.pose)
    pc.weight = 1.0

    # Orientation constraint
    oc = OrientationConstraint()
    oc.header = goal_pose_stamped.header
    oc.link_name = ee_link
    oc.orientation = goal_pose_stamped.pose.orientation
    oc.absolute_x_axis_tolerance = ang_tol
    oc.absolute_y_axis_tolerance = ang_tol
    oc.absolute_z_axis_tolerance = ang_tol
    oc.weight = 1.0

    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)
    return c

class InterceptPlanner:
    def __init__(self):
        self.object_topic = rospy.get_param("~object_topic", "/object/state")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.group_name = rospy.get_param("~group_name", "panda_arm")

        # Services
        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")
        self.plan_service = rospy.get_param("~plan_service", "/plan_kinematic_path")  # OMPL default
        # You can set ~plan_service:=/dgm/get_motion_plan to use your DGM service

        # Intercept search params
        self.latency_s = rospy.get_param("~latency_s", 0.15)     # sensor+compute+comm lag
        self.t_min = rospy.get_param("~t_min", 0.2)
        self.t_max = rospy.get_param("~t_max", 2.0)
        self.t_step = rospy.get_param("~t_step", 0.1)

        # Grasp offset in OBJECT frame (simple V1)
        # Approach the object from -X and keep same orientation as object
        self.grasp_offset_x = rospy.get_param("~grasp_offset_x", -0.12)
        self.grasp_offset_y = rospy.get_param("~grasp_offset_y", 0.0)
        self.grasp_offset_z = rospy.get_param("~grasp_offset_z", 0.0)

        # Goal tolerances
        self.pos_tol = rospy.get_param("~pos_tol", 0.02)
        self.ang_tol = rospy.get_param("~ang_tol", 0.20)

        # Planning params
        self.allowed_planning_time = rospy.get_param("~allowed_planning_time", 2.0)
        self.num_planning_attempts = int(rospy.get_param("~num_planning_attempts", 3))
        self.vel_scale = rospy.get_param("~vel_scale", 0.3)
        self.acc_scale = rospy.get_param("~acc_scale", 0.3)

        # Start state (deterministic & collision-free for Panda resources)
        self.start_state = panda_extended_open_start_state()

        self.last_odom = None

        self.pub_goal = rospy.Publisher("/intercept/goal_pose", PoseStamped, queue_size=1)
        self.pub_pred = rospy.Publisher("/intercept/object_pred", PoseStamped, queue_size=1)

        rospy.loginfo("Waiting for IK service: %s", self.ik_service)
        rospy.wait_for_service(self.ik_service, timeout=30.0)
        rospy.loginfo("Waiting for plan service: %s", self.plan_service)
        rospy.wait_for_service(self.plan_service, timeout=30.0)

        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)
        self.plan = rospy.ServiceProxy(self.plan_service, GetMotionPlan)

        rospy.Subscriber(self.object_topic, Odometry, self.cb_object, queue_size=1)

        rate_hz = rospy.get_param("~rate_hz", 2.0)
        self.timer = rospy.Timer(rospy.Duration(1.0 / rate_hz), self.on_timer)

        rospy.loginfo("InterceptPlanner ready. plan_service=%s ik_service=%s", self.plan_service, self.ik_service)

    def cb_object(self, msg: Odometry):
        self.last_odom = msg

    def predict_object_pose(self, odom: Odometry, t_future: float) -> PoseStamped:
        """
        Constant-velocity prediction of object pose in world frame.
        Orientation: hold constant in V1 (good enough to start).
        """
        p0 = odom.pose.pose.position
        v = odom.twist.twist.linear

        ps = PoseStamped()
        ps.header = odom.header
        ps.header.frame_id = odom.header.frame_id or self.world_frame
        ps.pose.position.x = p0.x + v.x * t_future
        ps.pose.position.y = p0.y + v.y * t_future
        ps.pose.position.z = p0.z + v.z * t_future

        ps.pose.orientation = odom.pose.pose.orientation  # V1: constant orientation
        return ps

    def make_grasp_goal(self, obj_pose: PoseStamped) -> PoseStamped:
        """
        V1 grasp goal: translate in object frame by fixed offset (ignoring rotation coupling).
        For now, apply offset in WORLD axes (simple). Next iteration: apply in object frame using quaternion rotation.
        """
        g = PoseStamped()
        g.header = Header(stamp=rospy.Time.now(), frame_id=obj_pose.header.frame_id)

        g.pose.position.x = obj_pose.pose.position.x + self.grasp_offset_x
        g.pose.position.y = obj_pose.pose.position.y + self.grasp_offset_y
        g.pose.position.z = obj_pose.pose.position.z + self.grasp_offset_z

        # Keep same orientation as object (V1)
        g.pose.orientation = obj_pose.pose.orientation
        return g

    def ik_feasible(self, goal_pose: PoseStamped) -> bool:
        req = GetPositionIKRequest()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.pose_stamped = goal_pose
        req.ik_request.robot_state = self.start_state
        req.ik_request.timeout = rospy.Duration(0.15)
        resp = self.ik(req)
        return resp.error_code.val == 1

    def choose_intercept_time(self, odom: Odometry):
        """
        Scan candidate times and pick the earliest feasible IK.
        """
        best = None
        t = self.t_min
        while t <= self.t_max + 1e-9:
            t_pred = t + self.latency_s
            obj_pred = self.predict_object_pose(odom, t_pred)
            goal = self.make_grasp_goal(obj_pred)

            self.pub_pred.publish(obj_pred)
            self.pub_goal.publish(goal)

            if self.ik_feasible(goal):
                best = (t, obj_pred, goal)
                break
            t += self.t_step
        return best

    def call_planner(self, goal: PoseStamped):
        mpr = MotionPlanRequest()
        mpr.group_name = self.group_name
        mpr.num_planning_attempts = self.num_planning_attempts
        mpr.allowed_planning_time = self.allowed_planning_time
        mpr.max_velocity_scaling_factor = self.vel_scale
        mpr.max_acceleration_scaling_factor = self.acc_scale
        mpr.start_state = self.start_state

        mpr.goal_constraints = [make_goal_constraints_from_pose(goal, self.ee_link,
                                                               pos_tol=self.pos_tol,
                                                               ang_tol=self.ang_tol)]
        req = GetMotionPlanRequest()
        req.motion_plan_request = mpr
        resp = self.plan(req)
        return resp.motion_plan_response

    def on_timer(self, _evt):
        if self.last_odom is None:
            return

        # Require correct frame
        if (self.last_odom.header.frame_id or "") == "":
            rospy.logwarn_throttle(2.0, "object odom has empty frame_id; expected %s", self.world_frame)

        choice = self.choose_intercept_time(self.last_odom)
        if choice is None:
            rospy.logwarn_throttle(1.0, "No feasible intercept found in [%.2f, %.2f] s", self.t_min, self.t_max)
            return

        t_hit, obj_pred, goal = choice
        rospy.loginfo("Intercept candidate: t_hit=%.2fs (latency=%.2fs) goal=(%.3f,%.3f,%.3f)",
                      t_hit, self.latency_s,
                      goal.pose.position.x, goal.pose.position.y, goal.pose.position.z)

        plan_resp = self.call_planner(goal)
        rospy.loginfo("Planner(%s) error_code=%d planning_time=%.3f",
                      self.plan_service, plan_resp.error_code.val, plan_resp.planning_time)

def main():
    rospy.init_node("intercept_planner")
    InterceptPlanner()
    rospy.spin()

if __name__ == "__main__":
    main()