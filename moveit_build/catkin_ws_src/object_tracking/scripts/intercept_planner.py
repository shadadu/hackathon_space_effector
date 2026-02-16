#!/usr/bin/env python3
import math
import rospy

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Vector3
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from shape_msgs.msg import SolidPrimitive

import tf2_ros


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
    c = Constraints()

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
        self.plan_service = rospy.get_param("~plan_service", "/plan_kinematic_path")  # or /dgm/get_motion_plan

        # Intercept search params
        self.latency_s = rospy.get_param("~latency_s", 0.15)     # sensor+compute+comm lag
        self.t_min = rospy.get_param("~t_min", 0.2)
        self.t_max = rospy.get_param("~t_max", 2.0)
        self.t_step = rospy.get_param("~t_step", 0.1)

        # Prediction toggles
        self.predict_orientation = rospy.get_param("~predict_orientation", True)

        # Simple reachability heuristic (stabilizes intercept selection)
        self.v_ee_max = rospy.get_param("~v_ee_max", 0.6)         # m/s (rough EE speed bound)
        self.w_dist = rospy.get_param("~w_dist", 1.0)
        self.w_time = rospy.get_param("~w_time", 0.05)
        self.top_k = int(rospy.get_param("~top_k_candidates", 5))

        # Grasp offset in OBJECT frame
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

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

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

    @staticmethod
    def quat_mul(q1, q2):
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return (
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        )

    @staticmethod
    def quat_from_omega_dt(wx, wy, wz, dt):
        theta = math.sqrt(wx*wx + wy*wy + wz*wz) * dt
        if theta < 1e-9:
            return (0.0, 0.0, 0.0, 1.0)
        n = math.sqrt(wx*wx + wy*wy + wz*wz)
        ax, ay, az = wx/n, wy/n, wz/n
        s = math.sin(theta/2.0)
        return (ax*s, ay*s, az*s, math.cos(theta/2.0))

    @staticmethod
    def quat_rotate(q, v):
        # q = (x,y,z,w), v=(vx,vy,vz)
        x, y, z, w = q
        vx, vy, vz = v
        # t = 2 * cross(q_vec, v)
        tx = 2.0 * (y * vz - z * vy)
        ty = 2.0 * (z * vx - x * vz)
        tz = 2.0 * (x * vy - y * vx)
        # v' = v + w*t + cross(q_vec, t)
        vpx = vx + w * tx + (y * tz - z * ty)
        vpy = vy + w * ty + (z * tx - x * tz)
        vpz = vz + w * tz + (x * ty - y * tx)
        return vpx, vpy, vpz

    def effective_prediction_dt(self, odom: Odometry, t_hit: float) -> float:
        """
        Use:
          dt = (now - msg_stamp) + latency + t_hit
        so prediction is correct even when messages are stale.
        """
        now = rospy.Time.now()
        msg_t = odom.header.stamp if odom.header.stamp != rospy.Time(0) else now
        age = (now - msg_t).to_sec()
        return max(0.0, age + self.latency_s + t_hit)

    def predict_object_pose(self, odom: Odometry, t_hit: float) -> PoseStamped:
        """
        Constant-velocity (and optional constant-omega) prediction in odom.header.frame_id.
        """
        dt = self.effective_prediction_dt(odom, t_hit)

        p0 = odom.pose.pose.position
        v = odom.twist.twist.linear

        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = odom.header.frame_id or self.world_frame

        ps.pose.position.x = p0.x + v.x * dt
        ps.pose.position.y = p0.y + v.y * dt
        ps.pose.position.z = p0.z + v.z * dt

        q0 = odom.pose.pose.orientation
        if self.predict_orientation:
            w = odom.twist.twist.angular
            dq = self.quat_from_omega_dt(w.x, w.y, w.z, dt)
            q = self.quat_mul((q0.x, q0.y, q0.z, q0.w), dq)
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        else:
            ps.pose.orientation = q0

        return ps

    def make_grasp_goal(self, obj_pose: PoseStamped) -> PoseStamped:
        """
        Apply grasp offset in OBJECT frame (rotated by object quaternion).
        """
        g = PoseStamped()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = obj_pose.header.frame_id

        q = obj_pose.pose.orientation
        ox, oy, oz = self.grasp_offset_x, self.grasp_offset_y, self.grasp_offset_z
        dx, dy, dz = self.quat_rotate((q.x, q.y, q.z, q.w), (ox, oy, oz))

        g.pose.position.x = obj_pose.pose.position.x + dx
        g.pose.position.y = obj_pose.pose.position.y + dy
        g.pose.position.z = obj_pose.pose.position.z + dz

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

    def ee_position_world(self):
        """
        Get EE position in world_frame using TF.
        """
        try:
            tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.ee_link, rospy.Time(0), rospy.Duration(0.2)
            )
            t = tf.transform.translation
            return (t.x, t.y, t.z)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "TF lookup failed (%s -> %s): %s", self.world_frame, self.ee_link, str(e))
            return None

    def choose_intercept_time(self, odom: Odometry):
        """
        Improved selection:
          1) scan times
          2) compute predicted goal
          3) apply simple reachability filter using v_ee_max
          4) score candidates and IK-check top_k
        """
        ee0 = self.ee_position_world()
        if ee0 is None:
            return None

        candidates = []
        t = self.t_min
        while t <= self.t_max + 1e-9:
            obj_pred = self.predict_object_pose(odom, t)
            goal = self.make_grasp_goal(obj_pred)

            # Publish for visualization/debug
            self.pub_pred.publish(obj_pred)
            self.pub_goal.publish(goal)

            # reachability heuristic (distance must be plausible)
            dx = goal.pose.position.x - ee0[0]
            dy = goal.pose.position.y - ee0[1]
            dz = goal.pose.position.z - ee0[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)

            if dist <= self.v_ee_max * max(t, 1e-3):
                score = self.w_dist * (dist*dist) + self.w_time * (t*t)
                candidates.append((score, t, obj_pred, goal))

            t += self.t_step

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        for score, t_hit, obj_pred, goal in candidates[:max(1, self.top_k)]:
            if self.ik_feasible(goal):
                return (t_hit, obj_pred, goal)

        return None

    def call_planner(self, goal: PoseStamped):
        mpr = MotionPlanRequest()
        mpr.group_name = self.group_name
        mpr.num_planning_attempts = self.num_planning_attempts
        mpr.allowed_planning_time = self.allowed_planning_time
        mpr.max_velocity_scaling_factor = self.vel_scale
        mpr.max_acceleration_scaling_factor = self.acc_scale
        mpr.start_state = self.start_state

        mpr.goal_constraints = [make_goal_constraints_from_pose(
            goal, self.ee_link, pos_tol=self.pos_tol, ang_tol=self.ang_tol
        )]

        req = GetMotionPlanRequest()
        req.motion_plan_request = mpr
        resp = self.plan(req)
        return resp.motion_plan_response

    def on_timer(self, _evt):
        if self.last_odom is None:
            return

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
