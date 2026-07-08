#!/usr/bin/env python3
import time
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import Point
from moveit_commander import MoveGroupCommander, RobotCommander
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from nav_msgs.msg import Odometry

from object_tracking.dgm_jax import load_checkpoint
from object_tracking.fk_client import FKClient
from object_tracking.micro_g_dgm_rollout import MicroGRolloutConfig, rollout_micro_g_dgm_policy


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def default_joint_vel_limits():
    return np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0], dtype=np.float64)


def odom_from_goal(goal_pos, frame_id):
    odom = Odometry()
    odom.header.stamp = rospy.Time.now()
    odom.header.frame_id = frame_id
    odom.child_frame_id = "object_link"
    odom.pose.pose.position = Point(float(goal_pos[0]), float(goal_pos[1]), float(goal_pos[2]))
    odom.pose.pose.orientation.w = 1.0
    return odom


class MicroGDGMPlannerService:
    def __init__(self):
        self.robot = RobotCommander()
        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.service_name = rospy.get_param("~service_name", "/micro_g_dgm/get_motion_plan")
        self.object_topic = rospy.get_param("~object_topic", "/object/state")
        self.object_max_age_s = float(rospy.get_param("~object_max_age_s", 0.50))
        self.service_wait_timeout = float(rospy.get_param("~service_wait_timeout", 30.0))

        self.T = float(rospy.get_param("~T", 2.0))
        self.dt = float(rospy.get_param("~dt", 0.02))
        self.R_q_diag = np.array(rospy.get_param("~R_q_diag", [0.15] * 7), dtype=np.float64)
        self.R_b_diag = np.array(rospy.get_param("~R_b_diag", [0.50] * 3), dtype=np.float64)
        self.joint_vel_limits = np.array(
            rospy.get_param("~joint_vel_limits", default_joint_vel_limits().tolist()),
            dtype=np.float64,
        )
        self.base_vel_limits = np.array(rospy.get_param("~base_vel_limits", [0.08, 0.08, 0.08]), dtype=np.float64)
        self.base_min = np.array(rospy.get_param("~base_min", [-0.50, -0.50, -0.20]), dtype=np.float64)
        self.base_max = np.array(rospy.get_param("~base_max", [0.50, 0.50, 0.50]), dtype=np.float64)
        self.base_position = np.array(rospy.get_param("~base_position", [0.0, 0.0, 0.0]), dtype=np.float64)

        self.jmin, self.jmax = panda_joint_limits()
        self.last_odom = None
        rospy.Subscriber(self.object_topic, Odometry, self.cb_object, queue_size=1)

        model_path = Path(rospy.get_param(
            "~model_path",
            "/root/catkin_ws/src/object_tracking/models/micro_g_dgm_v1.pkl",
        ))
        self.model = None
        if model_path.exists():
            self.model, meta = load_checkpoint(str(model_path))
            if int(meta.get("in_dim", 17)) != 17:
                rospy.logwarn("Loaded checkpoint in_dim=%s, expected 17 for micro-g DGM", meta.get("in_dim"))
            rospy.loginfo("Loaded micro-g DGM model: %s", model_path)
        else:
            rospy.logwarn("Micro-g DGM model not found at %s", model_path)

        fk_service = rospy.get_param("~fk_service", "/compute_fk")
        rospy.loginfo(
            "Waiting up to %.1fs for MoveIt FK service: %s",
            self.service_wait_timeout,
            fk_service,
        )
        try:
            self.fk = FKClient(
                service=fk_service,
                ee_link=self.ee_link,
                frame=self.world_frame,
                timeout=self.service_wait_timeout,
            )
        except rospy.ROSException as exc:
            raise rospy.ROSInitException(
                "Cannot start micro-g DGM planner without FK service '{}': {}".format(
                    fk_service, exc
                )
            )
        self.srv = rospy.Service(self.service_name, GetMotionPlan, self.handle)
        rospy.loginfo("Micro-g DGM planner service ready: %s", self.service_name)

    def cb_object(self, msg):
        self.last_odom = msg

    def latest_object_odom(self):
        if self.last_odom is None:
            return None
        stamp = self.last_odom.header.stamp if self.last_odom.header.stamp != rospy.Time(0) else rospy.Time.now()
        age = (rospy.Time.now() - stamp).to_sec()
        if 0.0 <= age <= self.object_max_age_s:
            return self.last_odom
        return None

    def extract_goal_position(self, mpr):
        pc = mpr.goal_constraints[0].position_constraints[0]
        goal_pose = pc.constraint_region.primitive_poses[0]
        return np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z], dtype=np.float64)

    def start_state_q0(self, mpr, active_joints):
        if mpr.start_state and mpr.start_state.joint_state.name:
            s_map = dict(zip(mpr.start_state.joint_state.name, mpr.start_state.joint_state.position))
            return np.array([s_map[j] for j in active_joints], dtype=np.float64)
        st = self.robot.get_current_state()
        s_map = dict(zip(st.joint_state.name, st.joint_state.position))
        return np.array([s_map[j] for j in active_joints], dtype=np.float64)

    def handle(self, req):
        mpr = req.motion_plan_request
        resp = MotionPlanResponse()
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        resp.planning_time = 0.0

        if self.model is None:
            resp.error_code.val = MoveItErrorCodes.ROBOT_STATE_STALE
            return GetMotionPlanResponse(motion_plan_response=resp)
        if not mpr.goal_constraints:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        group = MoveGroupCommander(mpr.group_name or self.group_name)
        active_joints = group.get_active_joints()
        if len(active_joints) != 7:
            resp.error_code.val = MoveItErrorCodes.INVALID_GROUP_NAME
            return GetMotionPlanResponse(motion_plan_response=resp)

        q0 = clamp(self.start_state_q0(mpr, active_joints), self.jmin, self.jmax)
        goal_pos = self.extract_goal_position(mpr)
        object_odom = self.latest_object_odom()
        if object_odom is None:
            object_odom = odom_from_goal(goal_pos, self.world_frame)

        cfg = MicroGRolloutConfig(
            T=self.T,
            dt=self.dt,
            joint_min=self.jmin,
            joint_max=self.jmax,
            joint_vel_limits=self.joint_vel_limits,
            base_vel_limits=self.base_vel_limits,
            R_q_diag=self.R_q_diag,
            R_b_diag=self.R_b_diag,
            base_min=self.base_min,
            base_max=self.base_max,
            grasp_pos_tol=float(rospy.get_param("~grasp_pos_tol", 0.08)),
            grasp_vel_tol=float(rospy.get_param("~grasp_vel_tol", 0.05)),
            max_nan_guard=int(rospy.get_param("~max_nan_guard", 5)),
        )

        t0 = time.time()
        try:
            traj, _, b_hist, r_hist = rollout_micro_g_dgm_policy(
                model=self.model,
                q0=q0,
                b0=self.base_position,
                object_odom=object_odom,
                active_joints=active_joints,
                group=group,
                cfg=cfg,
                fk_client=self.fk,
                object_state_provider=self.latest_object_odom,
            )
        except Exception as exc:
            rospy.logerr("Micro-g DGM rollout failed: %s", exc)
            resp.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            return GetMotionPlanResponse(motion_plan_response=resp)

        if not traj.joint_trajectory.points:
            resp.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            return GetMotionPlanResponse(motion_plan_response=resp)

        self.base_position = b_hist[-1].copy()
        resp.trajectory = traj
        resp.planning_time = float(time.time() - t0)
        rospy.loginfo("Micro-g DGM final relative position: %s", r_hist[-1])
        return GetMotionPlanResponse(motion_plan_response=resp)


def main():
    rospy.init_node("micro_g_dgm_planner_node")
    MicroGDGMPlannerService()
    rospy.spin()


if __name__ == "__main__":
    main()
