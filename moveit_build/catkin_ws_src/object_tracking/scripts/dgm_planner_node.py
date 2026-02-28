#!/usr/bin/env python3
import os
import time
import numpy as np
import rospy

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import RobotCommander, MoveGroupCommander

# Optional Jacobian service hook
from jacobian_server.srv import GetJacobian, GetJacobianRequest

# from catkin_ws_src.object_tracking.scripts.dgm_model import load_model, DGMValueNet

from object_tracking.dgm_model import DGMValueNet
# from object_tracking.trajectory_executor_manager import TrajectoryExecutorManager


def decode(code):
    for k, v in MoveItErrorCodes.__dict__.items():
        if isinstance(v, int) and v == code:
            return k
    return str(code)
# from dgm_model import load_model, DGMValueNet
# from dgm_rollout import RolloutConfig, rollout_dgm_joint_policy

# try:
#     from object_tracking.dgm_model import load_model, DGMValueNet
#     from object_tracking.dgm_rollout import RolloutConfig, rollout_dgm_joint_policy
# except ImportError:
#     # fallback if running directly from source without proper PYTHONPATH
#     import os, sys
#     this_dir = os.path.dirname(os.path.abspath(__file__))
#     pkg_src = os.path.abspath(os.path.join(this_dir, "..", "src"))
#     if pkg_src not in sys.path:
#         sys.path.insert(0, pkg_src)
#     from object_tracking.dgm_model import load_model, DGMValueNet
#     from object_tracking.dgm_rollout import RolloutConfig, rollout_dgm_joint_policy


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float64)
    return jmin, jmax


def default_vel_limits():
    # Conservative joint velocity limits (rad/s) for stable rollout
    return np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0], dtype=np.float64)


class DGMPlannerService:
    def __init__(self):
        self.robot = RobotCommander()
        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.world_frame = rospy.get_param("~world_frame", "world")

        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")
        self.service_name = rospy.get_param("~service_name", "/dgm/get_motion_plan")

        # DGM model
        self.model_path = rospy.get_param("~model_path", "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pt")
        self.device = rospy.get_param("~device", "cpu")
        self.model = None  # type: DGMValueNet

        # Rollout config (your requested values)
        self.T = float(rospy.get_param("~T", 2.0))
        self.dt = float(rospy.get_param("~dt", 0.02))
        self.R_diag = np.array(rospy.get_param("~R_diag", [0.15]*7), dtype=np.float64)
        self.vel_limits = np.array(rospy.get_param("~vel_limits", default_vel_limits().tolist()), dtype=np.float64)

        self.jmin, self.jmax = panda_joint_limits()

        # Optional Jacobian hook (future guidance / regularizer)
        self.jacobian_service = rospy.get_param("~jacobian_service", "/get_jacobian")
        self.use_jacobian_hook = bool(rospy.get_param("~use_jacobian_hook", False))
        self.jac = None

        # IK proxy (optional check)
        rospy.wait_for_service(self.ik_service, timeout=60.0)
        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)

        if os.path.exists(self.model_path):
            self.model = DGMValueNet.load_model(self.model_path, device=self.device)
            rospy.loginfo("Loaded DGM model: %s", self.model_path)
        else:
            rospy.logwarn("DGM model not found at %s. Planner will return INVALID_MOTION_PLAN.", self.model_path)

        if self.use_jacobian_hook:
            rospy.wait_for_service(self.jacobian_service, timeout=60.0)
            self.jac = rospy.ServiceProxy(self.jacobian_service, GetJacobian)
            rospy.loginfo("Jacobian hook enabled: %s", self.jacobian_service)

        self.srv = rospy.Service(self.service_name, GetMotionPlan, self.handle)
        rospy.loginfo("DGM planner service up: %s (IK: %s)", self.service_name, self.ik_service)

    def extract_goal_position(self, mpr):
        # Expect goal constraint created from Pose constraint in benchmark/intercept planners
        pc = mpr.goal_constraints[0].position_constraints[0]
        goal_pose = pc.constraint_region.primitive_poses[0]
        return np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z], dtype=np.float64)

    def start_state_q0(self, mpr, active_joints):
        # Prefer provided start_state
        if mpr.start_state and mpr.start_state.joint_state.name:
            s_map = dict(zip(mpr.start_state.joint_state.name, mpr.start_state.joint_state.position))
            return np.array([s_map[j] for j in active_joints], dtype=np.float64)

        # Else try current state from robot commander
        st = self.robot.get_current_state()
        s_map = dict(zip(st.joint_state.name, st.joint_state.position))
        return np.array([s_map[j] for j in active_joints], dtype=np.float64)

    def ik_feasible_pose(self, mpr) -> bool:
        # Optional: check that goal pose is IK-feasible (helps early exit)
        try:
            pc = mpr.goal_constraints[0].position_constraints[0]
            goal_pose = pc.constraint_region.primitive_poses[0]
            goal_frame = pc.header.frame_id if pc.header.frame_id else self.world_frame

            ikreq = GetPositionIKRequest()
            ikreq.ik_request.group_name = mpr.group_name or self.group_name
            ikreq.ik_request.ik_link_name = self.ee_link
            ikreq.ik_request.pose_stamped.header.frame_id = goal_frame
            ikreq.ik_request.pose_stamped.pose = goal_pose
            ikreq.ik_request.robot_state = mpr.start_state if (mpr.start_state and mpr.start_state.joint_state.name) else self.robot.get_current_state()
            ikreq.ik_request.timeout = rospy.Duration(0.2)
            ikresp = self.ik(ikreq)
            return ikresp.error_code.val == MoveItErrorCodes.SUCCESS
        except Exception:
            return False

    def jacobian_hook_call(self, active_joints, q):
        # Hook for future: get Jacobian for debugging/regularization
        if not self.use_jacobian_hook or self.jac is None:
            return
        try:
            req = GetJacobianRequest()
            req.group_name = self.group_name
            req.link_name = self.ee_link
            req.joint_names = list(active_joints)
            req.joint_positions = [float(x) for x in q.tolist()]
            # reference point at EE origin
            req.reference_point.x = 0.0
            req.reference_point.y = 0.0
            req.reference_point.z = 0.0
            _ = self.jac(req)
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Jacobian hook call failed: %s", str(e))

    def handle(self, req):
        mpr = req.motion_plan_request

        resp = MotionPlanResponse()
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        resp.planning_time = 0.0

        if self.model is None:
            resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
            return GetMotionPlanResponse(motion_plan_response=resp)

        if not mpr.goal_constraints:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Optional early IK feasibility check (can disable if you want pure DGM)
        if not bool(rospy.get_param("~skip_ik_check", False)):
            if not self.ik_feasible_pose(mpr):
                resp.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                return GetMotionPlanResponse(motion_plan_response=resp)

        group = MoveGroupCommander(mpr.group_name or self.group_name)
        active_joints = group.get_active_joints()
        if len(active_joints) != 7:
            resp.error_code.val = MoveItErrorCodes.INVALID_GROUP_NAME
            return GetMotionPlanResponse(motion_plan_response=resp)

        goal_pos = self.extract_goal_position(mpr)
        q0 = self.start_state_q0(mpr, active_joints)

        # Rollout
        cfg = RolloutConfig(
            T=self.T,
            dt=self.dt,
            vel_limits=self.vel_limits,
            joint_min=self.jmin,
            joint_max=self.jmax,
            R_diag=self.R_diag,
        )

        t0 = time.time()
        traj, q_hist = rollout_dgm_joint_policy(
            model=self.model,
            q0=q0,
            goal_pos=goal_pos,
            active_joints=active_joints,
            cfg=cfg,
            device=self.device,
        )
        resp.planning_time = float(time.time() - t0)

        # Jacobian hook (one call at start/end for now)
        self.jacobian_hook_call(active_joints, q_hist[0])
        self.jacobian_hook_call(active_joints, q_hist[-1])

        resp.trajectory = traj
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        return GetMotionPlanResponse(motion_plan_response=resp)


if __name__ == "__main__":
    rospy.init_node("dgm_planner_node")
    DGMPlannerService()
    rospy.spin()