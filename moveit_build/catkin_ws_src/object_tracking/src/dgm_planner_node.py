#!/usr/bin/env python3
import os
import time
import numpy as np
# from dgm_model import DGMValueNet
import rospy
import torch
from torch import optim

from pathlib import Path
import stat

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import RobotCommander, MoveGroupCommander
from moveit_msgs.msg import RobotTrajectory, RobotState
from sensor_msgs.msg import JointState

# Optional Jacobian service hook
from jacobian_server.srv import GetJacobian, GetJacobianRequest

from object_tracking.dgm_model import DGMValueNet
# from trajectory_executor_manager import get_panda_start_pose
from object_tracking.dgm_rollout import RolloutConfig, rollout_dgm_joint_policy, rollout_value_policy
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

from trajectory_executor_manager import panda_extended_open_start_state

import moveit_commander

from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest

def get_ee_translation():
    rospy.wait_for_service('compute_fk')
    fk_srv = rospy.ServiceProxy('compute_fk', GetPositionFK)
    # Create the FK Request
    request = GetPositionFKRequest()
    request.fk_link_names = ["panda_hand"]
    try:
        response = fk_srv(request)
        if response.error_code.val == 1:  # SUCCESS
            translation = response.pose_stamped[0].pose.position
            orientation = response.pose_stamped[0].pose.orientation
            print(f"Final EE Position: x={translation.x}, y={translation.y}, z={translation.z}")
            return translation, orientation
    except rospy.ServiceException as e:
        rospy.logerr("FK service call failed: %s" % e)
def get_panda_start_pose(request):
    rospy.wait_for_service('compute_fk')
    fk_srv = rospy.ServiceProxy('compute_fk', GetPositionFK)

    fk_request = GetPositionFKRequest()
    # Specify the link you want the coordinates for
    fk_request.fk_link_names = ["panda_hand"]
    fk_request.header.frame_id = "panda_link0"  # Usually the base frame
    fk_request.robot_state = request.start_state

    try:
        response = fk_srv(fk_request)
        if response.error_code.val == 1:
            pose = response.pose_stamped[0].pose

            # Cartesian Coordinates
            pos = pose.position
            # Quaternion Coordinates
            ori = pose.orientation

            print(f"Start Position: x={pos.x}, y={pos.y}, z={pos.z}")
            print(f"Start Orientation (Quat): x={ori.x}, y={ori.y}, z={ori.z}, w={ori.w}")
            return pose
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s" % e)

def get_joint_state_translation(joint_state):
    rospy.wait_for_service('compute_fk')
    fk_srv = rospy.ServiceProxy('compute_fk', GetPositionFK)

    # last_point = plan.trajectory.joint_trajectory.points[-1]

    # Build the RobotState message
    robot_state = RobotState()
    # robot_state.joint_state.name = plan.trajectory.joint_trajectory.joint_names
    robot_state.joint_state = joint_state

    # Create the FK Request
    request = GetPositionFKRequest()
    request.fk_link_names = ["panda_hand"]
    request.robot_state = robot_state

    try:
        response = fk_srv(request)
        if response.error_code.val == 1:  # SUCCESS
            translation = response.pose_stamped[0].pose.position
            orientation = response.pose_stamped[0].pose.orientation
            print(f"Joint State EE Position: x={translation.x}, y={translation.y}, z={translation.z}")
            return translation, orientation
    except rospy.ServiceException as e:
        rospy.logerr("FK service call failed: %s" % e)

def decode(code):
    for k, v in MoveItErrorCodes.__dict__.items():
        if isinstance(v, int) and v == code:
            return k
    return str(code)


def load_model(path: Path, hidden: int, depth: int, lr: float, device: str = "cpu") -> DGMValueNet:
    model = DGMValueNet(in_dim=11, hidden=hidden, depth=depth).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    checkpoint = torch.load(str(path.resolve()))
    # Load the model state dictionary from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()
    return model


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def default_vel_limits():
    # Conservative joint velocity limits (rad/s) for stable rollout
    return np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0], dtype=np.float64)


def finite(x: np.ndarray) -> bool:
    return np.all(np.isfinite(x))


def compute_derivatives(q_hist: np.ndarray, dt: float):
    # q_hist: (N,7)
    qdot = np.diff(q_hist, axis=0) / dt  # (N-1,7)
    qdd = np.diff(qdot, axis=0) / dt  # (N-2,7)
    qjerk = np.diff(qdd, axis=0) / dt  # (N-3,7)
    return qdot, qdd, qjerk


def enforce_time_monotone(traj: RobotTrajectory, dt: float):
    # rewrite time_from_start to be monotone and consistent with dt
    for k, pt in enumerate(traj.joint_trajectory.points):
        pt.time_from_start = rospy.Duration.from_sec(k * dt)


def ensure_joint_dims(traj: RobotTrajectory, n_joints: int):
    jn = traj.joint_trajectory.joint_names
    for k, pt in enumerate(traj.joint_trajectory.points):
        if len(pt.positions) != n_joints:
            raise RuntimeError(f"Waypoint {k}: positions len={len(pt.positions)} != {n_joints}")
        # velocities are optional in MoveIt trajectories, but if present must match
        if pt.velocities and len(pt.velocities) != n_joints:
            raise RuntimeError(f"Waypoint {k}: velocities len={len(pt.velocities)} != {n_joints}")


def robot_state_from_q(active_joints, q: np.ndarray) -> RobotState:
    rs = RobotState()
    js = JointState()
    js.name = list(active_joints)
    js.position = [float(x) for x in q.tolist()]
    rs.joint_state = js
    return rs


def check_limits(q_hist, jmin, jmax, vel_limits, dt,
                 acc_limits=None, jerk_limits=None):
    """
    Returns (ok:bool, reason:str, metrics:dict)
    """
    if not finite(q_hist):
        return False, "non_finite_q", {}

    # position
    if np.any(q_hist < (jmin[None, :] - 1e-9)) or np.any(q_hist > (jmax[None, :] + 1e-9)):
        return False, "position_limit_violation", {}

    qdot, qdd, qjerk = compute_derivatives(q_hist, dt)

    if not finite(qdot):
        return False, "non_finite_qdot", {}

    # velocity (∞-norm per joint)
    vmax = np.max(np.abs(qdot), axis=0)
    if np.any(vmax > (vel_limits + 1e-6)):
        return False, f"velocity_limit_violation vmax={vmax}", {"vmax": vmax}

    # if we don't have explicit acc/jerk limits yet, we still compute them for logging/penalty
    amax = np.max(np.abs(qdd), axis=0) if qdd.shape[0] else np.zeros(7)
    jmaxv = np.max(np.abs(qjerk), axis=0) if qjerk.shape[0] else np.zeros(7)

    if acc_limits is not None and qdd.shape[0]:
        if np.any(amax > (acc_limits + 1e-6)):
            return False, f"acc_limit_violation amax={amax}", {"amax": amax}

    if jerk_limits is not None and qjerk.shape[0]:
        if np.any(jmaxv > (jerk_limits + 1e-6)):
            return False, f"jerk_limit_violation jmax={jmaxv}", {"jmax": jmaxv}

    return True, "ok", {"vmax": vmax, "amax": amax, "jmax": jmaxv}


def validate_with_moveit_state_validity(
        svc: rospy.ServiceProxy,
        active_joints,
        q_hist: np.ndarray,
        group_name: str,
        stride: int = 5
):
    """
    Subsample trajectory states and call /check_state_validity.
    Returns (ok, first_bad_index, message)
    """
    req = GetStateValidityRequest()
    req.group_name = group_name

    N = q_hist.shape[0]
    for k in range(0, N, max(1, stride)):
        req.robot_state = robot_state_from_q(active_joints, q_hist[k])
        try:
            resp = svc(req)
        except Exception as e:
            return False, k, f"state_validity_call_failed: {e}"
        # resp.valid is bool in MoveIt
        if not resp.valid:
            return False, k, "collision_or_constraints_invalid"
    # rospy.loginfo("MoveIt state validity passed", req.robot_state )
    return True, -1, "ok"


class DGMPlannerService:
    def __init__(self):
        rospy.loginfo("DGM startup before init EE coordinates before init attempt = %s", get_ee_translation())
        self.robot = RobotCommander()
        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.world_frame = rospy.get_param("~world_frame", "world")

        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")
        self.service_name = rospy.get_param("~service_name", "/dgm/get_motion_plan")
        rospy.loginfo("DGM startup after ik EE coordinates before init attempt = %s", get_ee_translation())

        # DGM model
        self.model_path = rospy.get_param("~model_path", "/root/catkin_ws/src/objecttracking/models/panda_dgm_v1.pth")
        self.device = rospy.get_param("~device", "cpu")
        self.model = None  # type: DGMValueNet

        # Rollout config
        self.T = float(rospy.get_param("~T", 2.0))
        self.dt = float(rospy.get_param("~dt", 0.02))
        self.R_diag = np.array(rospy.get_param("~R_diag", [0.15] * 7), dtype=np.float64)
        self.vel_limits = np.array(rospy.get_param("~vel_limits", default_vel_limits().tolist()), dtype=np.float64)

        self.jmin, self.jmax = panda_joint_limits()

        # Optional Jacobian hook (future guidance / regularizer)
        self.jacobian_service = rospy.get_param("~jacobian_service", "/get_jacobian")
        self.use_jacobian_hook = bool(rospy.get_param("~use_jacobian_hook", False))
        self.jac = None

        # IK proxy (optional check)
        rospy.wait_for_service(self.ik_service, timeout=60.0)
        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)
        rospy.loginfo("DGM startup ik proxy before init EE coordinates before init attempt = %s", get_ee_translation())

        hidden = int(rospy.get_param("~hidden", 256))
        depth = int(rospy.get_param("~depth", 4))
        mdl_path = "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pth"
        path = Path(mdl_path)
        # Grant owner read/write, and others read (644)
        path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        rospy.loginfo(f"path is a file {path.is_file()}")
        rospy.loginfo(f"path is accessible {os.access(path, os.X_OK)}")
        if Path.exists(path):
            self.model = load_model(path=path, hidden=hidden, depth=depth, lr=3e-4, device=self.device)
            rospy.loginfo("Loaded DGM model: %s", mdl_path)
        else:
            # rospy.loginfo(f"File path exists? {os.path.exists()}")
            rospy.loginfo("DGM model not found at %s. Planner will return ROBOT_STATE_STALE.", mdl_path)
            # raise Exception(f"DGM model not found or not loaded via path {mdl_path}. Planner will return ROBOT_STATE_STALE.")

        if self.use_jacobian_hook:
            rospy.wait_for_service(self.jacobian_service, timeout=60.0)
            self.jac = rospy.ServiceProxy(self.jacobian_service, GetJacobian)
            rospy.loginfo("Jacobian hook enabled: %s", self.jacobian_service)

        # Optional physical initialization of Panda before serving plans
        # init_ok = self.opt_initialize_robot_pose()
        init_ee_coords = get_ee_translation()
        rospy.loginfo("DGM startup EE coordinates before init attempt = %s", init_ee_coords)

        init_ok = self.move_to_ready_joint_pose()

        # Report actual startup EE coordinates after initialization attempt
        ee_coords = get_ee_translation()
        rospy.loginfo("DGM startup EE coordinates = %s", ee_coords)

        self.srv = rospy.Service(self.service_name, GetMotionPlan, self.handle)
        rospy.loginfo("DGM planner service up: %s (IK: %s, init_ok=%s)",
                      self.service_name, self.ik_service, str(init_ok))

    def move_ee_to_xyz(self, x, y, z, quat=None, use_current_orientation=True):
        """
        Physically move the Panda end effector to (x,y,z) using MoveIt.
        ROS 1 / MoveIt 1 version.

        Parameters
        ----------
        x, y, z : float
            Target EE position in self.world_frame.
        quat : tuple or None
            Optional quaternion (qx, qy, qz, qw). Used only if
            use_current_orientation is False.
        use_current_orientation : bool
            If True, keep current EE orientation and only change XYZ.
        """
        group = MoveGroupCommander(self.group_name)
        group.set_pose_reference_frame(self.world_frame)
        group.set_end_effector_link(self.ee_link)

        group.set_planning_time(float(rospy.get_param("~init_planning_time", 5.0)))
        group.set_num_planning_attempts(int(rospy.get_param("~init_num_attempts", 5)))
        group.set_max_velocity_scaling_factor(float(rospy.get_param("~init_vel_scale", 0.25)))
        group.set_max_acceleration_scaling_factor(float(rospy.get_param("~init_acc_scale", 0.25)))

        pose_goal = group.get_current_pose(self.ee_link).pose

        pose_goal.position.x = float(x)
        pose_goal.position.y = float(y)
        pose_goal.position.z = float(z)

        if not use_current_orientation:
            if quat is None:
                raise ValueError("quat must be provided when use_current_orientation=False")
            pose_goal.orientation.x = float(quat[0])
            pose_goal.orientation.y = float(quat[1])
            pose_goal.orientation.z = float(quat[2])
            pose_goal.orientation.w = float(quat[3])

        group.set_start_state_to_current_state()
        group.set_pose_target(pose_goal, self.ee_link)

        rospy.loginfo("Initializing Panda EE to x=%.3f y=%.3f z=%.3f in frame %s",
                      x, y, z, self.world_frame)

        success = group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

        if success:
            rospy.loginfo("Initialization move succeeded.")
        else:
            rospy.logwarn("Initialization move failed.")

        return success

    def move_to_ready_joint_pose(self):
        group = MoveGroupCommander(self.group_name)
        group.set_pose_reference_frame(self.world_frame)
        group.set_end_effector_link(self.ee_link)

        group.set_planning_time(5.0)
        group.set_num_planning_attempts(5)
        group.set_max_velocity_scaling_factor(0.25)
        group.set_max_acceleration_scaling_factor(0.25)

        ready = [0.0, 0.0, 0.0, 0.0, 0.0, 1.571, 0.785]
        group.set_start_state_to_current_state()
        group.set_joint_value_target(ready)

        success = group.go(wait=True)
        group.stop()
        group.clear_pose_targets()

        if success:
            pose = group.get_current_pose(self.ee_link).pose
            rospy.loginfo("Ready pose EE position: x=%.3f y=%.3f z=%.3f",
                          pose.position.x, pose.position.y, pose.position.z)
        return success

    def maybe_initialize_robot_pose(self):
        initialize_on_start = bool(rospy.get_param("~initialize_on_start", True))
        if not initialize_on_start:
            return True

        use_ready_joint_pose = bool(rospy.get_param("~init_use_ready_joint_pose", True))
        if use_ready_joint_pose:
            ok = self.move_to_ready_joint_pose()
            if ok:
                return True
            rospy.logwarn("Ready joint pose failed, falling back to Cartesian init.")

        init_x = float(rospy.get_param("~init_x", 0.40))
        init_y = float(rospy.get_param("~init_y", 0.00))
        init_z = float(rospy.get_param("~init_z", 0.40))
        return self.move_ee_to_xyz(init_x, init_y, init_z, use_current_orientation=True)

    def opt_initialize_robot_pose(self):
        """
        Optional startup initialization of the Panda EE pose.
        Controlled by ROS params.

        Params
        ------
        ~initialize_on_start : bool
        ~init_x, ~init_y, ~init_z : float
        ~init_use_current_orientation : bool
        ~init_qx, ~init_qy, ~init_qz, ~init_qw : float
        """
        initialize_on_start = bool(rospy.get_param("~initialize_on_start", True))
        if not initialize_on_start:
            rospy.loginfo("Startup EE initialization disabled.")
            return True
        # A: x = 0.35, y = 0.00, z = 0.45
        # B: x = 0.40, y = 0.00, z = 0.40
        # C: x = 0.45, y = 0.00, z = 0.35
        # D: x = 0.40, y = 0.15, z = 0.40
        # E: x = 0.40, y = -0.15, z = 0.40
        init_x = float(rospy.get_param("~init_x", 0.35))
        init_y = float(rospy.get_param("~init_y", 0.00))
        init_z = float(rospy.get_param("~init_z", 0.45))
        use_current_orientation = bool(rospy.get_param("~init_use_current_orientation", True))

        quat = None
        if not use_current_orientation:
            quat = (
                float(rospy.get_param("~init_qx", 0.0)),
                float(rospy.get_param("~init_qy", 0.0)),
                float(rospy.get_param("~init_qz", 0.0)),
                float(rospy.get_param("~init_qw", 1.0)),
            )

        rospy.sleep(1.0)  # let MoveIt state settle
        ok = self.move_ee_to_xyz(
            init_x, init_y, init_z,
            quat=quat,
            use_current_orientation=use_current_orientation
        )

        if ok:
            ee_pose = get_ee_translation()
            rospy.loginfo("Post-init EE pose: %s", ee_pose)
        else:
            rospy.logwarn("Robot initialization was requested but did not succeed.")

        return ok
    def extract_goal_position(self, mpr):
        # Expect goal constraint created from Pose constraint in benchmark/intercept planners
        pc = mpr.goal_constraints[0].position_constraints[0]
        goal_pose = pc.constraint_region.primitive_poses[0]
        return np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z], dtype=np.float64)

    # def extract_start_position(self, mpr):
    #     # Expect goal constraint created from Pose constraint in benchmark/intercept planners
    #     pc = mpr.start_state.joint_state.position.position_constraints[0]
    #     goal_pose = pc.constraint_region.primitive_poses[0]
    #     return np.array([goal_pose.position.x, goal_pose.position.y, goal_pose.position.z], dtype=np.float64)

    def start_state_q0(self, mpr, active_joints):
        # Prefer provided start_state
        if mpr.start_state and mpr.start_state.joint_state.name:
            s_map = dict(zip(mpr.start_state.joint_state.name, mpr.start_state.joint_state.position))
            rospy.loginfo("Loaded start state from Motion Plan Request")
            rospy.loginfo("Joint state position: %s", mpr.start_state.joint_state.position)
            trans, ori = get_joint_state_translation(mpr.start_state.joint_state)
            return np.array([s_map[j] for j in active_joints], dtype=np.float64)

        # Else try current state from robot commander
        rospy.loginfo("No start state provided by mpr; loading current robot state as start state")
        st = self.robot.get_current_state()
        get_joint_state_translation(st.joint_state)
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
            ikreq.ik_request.robot_state = mpr.start_state if (
                        mpr.start_state and mpr.start_state.joint_state.name) else self.robot.get_current_state()
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
        mpr_robot_state = mpr.start_state
        # ee_start_pos = get_panda_start_pose(mpr_start_state)
        rospy.loginfo("Handle mpr_robot_state = %s", str(mpr_robot_state))
        rospy.loginfo("Handle mpr ee_start_pos = %s", str(mpr))

        resp = MotionPlanResponse()
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        resp.planning_time = 0.0

        # ---- basic request checks ----
        if not mpr.goal_constraints:
            rospy.loginfo("Invalid goal constraints")
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        if self.model is None:
            rospy.logerr("DGM model is not loaded; returning ROBOT_STATE_STALE %s", str(self.model))
            resp.error_code.val = MoveItErrorCodes.ROBOT_STATE_STALE
            # raise Exception("DGMValueNet model not loaded")
            return GetMotionPlanResponse(motion_plan_response=resp)

        # ---- group joints ----
        group = MoveGroupCommander(mpr.group_name or self.group_name)
        active_joints = group.get_active_joints()
        if len(active_joints) != 7:
            rospy.logerr("Expected 7 active joints, got %d (%s)", len(active_joints), active_joints)
            resp.error_code.val = MoveItErrorCodes.INVALID_GROUP_NAME
            return GetMotionPlanResponse(motion_plan_response=resp)

        # ---- goal + start ----
        goal_pos = self.extract_goal_position(mpr)
        q0 = self.start_state_q0(mpr, active_joints)
        q0 = np.minimum(np.maximum(q0, self.jmin), self.jmax)

        # optional: early IK feasibility (helps reject impossible goals quickly)
        if not bool(rospy.get_param("~skip_ik_check", False)):
            if not self.ik_feasible_pose(mpr):
                rospy.logwarn("Pose infeasible")
                resp.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                return GetMotionPlanResponse(motion_plan_response=resp)

        # ---- rollout ----
        cfg = RolloutConfig(
            T=self.T,
            dt=float(self.dt),
            vel_limits=self.vel_limits,
            joint_min=self.jmin,
            joint_max=self.jmax,
            R_diag=self.R_diag,
            max_nan_guard=int(rospy.get_param("~max_nan_guard", 5)),
        )
        rospy.loginfo("DGM rollout input q0=%s, active_joints=%s", q0, active_joints)
        rospy.loginfo("DGM goal_pos= %s", goal_pos)
        rospy.loginfo("jmin and jmax: %s, %s", self.jmin, self.jmax)
        rospy.loginfo("mpr start state: %s ", mpr.start_state.joint_state.position)
        ee_pos = get_ee_translation()
        rospy.loginfo("start state ee pos: %s ", ee_pos)
        t0 = time.time()
        try:
            traj, q_hist = rollout_dgm_joint_policy(
                model=self.model,
                q0=q0,
                goal_pos=goal_pos,
                active_joints=active_joints,
                cfg=cfg,
                device=self.device,
            )
        except Exception as e:
            rospy.logerr("DGM rollout failed: %s", str(e))
            resp.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            return GetMotionPlanResponse(motion_plan_response=resp)

        resp.planning_time = float(time.time() - t0)

        # ---- I.5 enforce time monotonicity ----
        try:
            enforce_time_monotone(traj, self.dt)
            ensure_joint_dims(traj, n_joints=len(active_joints))
        except Exception as e:
            rospy.logerr("Trajectory formatting invalid: %s", str(e))
            resp.error_code.val = MoveItErrorCodes.CONTROL_FAILED
            return GetMotionPlanResponse(motion_plan_response=resp)

        # ---- I.4 start consistency ----
        # Ensure first point matches q0 (small numerical tolerance)
        if np.max(np.abs(np.array(traj.joint_trajectory.points[0].positions) - q0)) > 1e-6:
            rospy.logwarn("First waypoint != start_state; forcing first waypoint to q0")
            traj.joint_trajectory.points[0].positions = [float(x) for x in q0.tolist()]
            # Also adjust q_hist[0] to match for downstream checks
            q_hist[0, :] = q0

        # ---- I.1 limits + continuity/jerk metrics ----
        ok_limits, reason, met = check_limits(
            q_hist=q_hist,
            jmin=self.jmin,
            jmax=self.jmax,
            vel_limits=self.vel_limits,
            dt=self.dt,
            acc_limits=None,  # you can add later
            jerk_limits=None,  # you can add later
        )
        if not ok_limits:
            rospy.logwarn("DGM plan rejected by limit checks: %s", reason)
            resp.error_code.val = MoveItErrorCodes.PREEMPTED
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Log jerk / smoothness stats (useful for training penalties)
        jerk = met.get("jmax", None)
        if jerk is not None:
            rospy.loginfo("DGM smoothness: vmax=%s amax=%s jmax=%s",
                          np.array2string(met["vmax"], precision=2),
                          np.array2string(met["amax"], precision=2),
                          np.array2string(met["jmax"], precision=2))

        # ---- I.2 collision/state validity ----
        try:
            rospy.wait_for_service("/check_state_validity", timeout=2.0)
            state_validity = rospy.ServiceProxy("/check_state_validity", GetStateValidity)
        except Exception:
            # try common namespaced form
            try:
                rospy.wait_for_service("/move_group/check_state_validity", timeout=2.0)
                state_validity = rospy.ServiceProxy("/move_group/check_state_validity", GetStateValidity)
            except Exception as e:
                rospy.logwarn("No state validity service reachable; skipping collision validation (%s)", str(e))
                state_validity = None

        if state_validity is not None:
            stride = int(rospy.get_param("~validity_stride", 5))  # check every 5th point by default
            ok_val, bad_k, msg = validate_with_moveit_state_validity(
                svc=state_validity,
                active_joints=active_joints,
                q_hist=q_hist,
                group_name=(mpr.group_name or self.group_name),
                stride=stride,
            )
            if not ok_val:
                rospy.logwarn("DGM plan rejected by MoveIt validity at k=%d: %s", bad_k, msg)
                resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
                return GetMotionPlanResponse(motion_plan_response=resp)

        # ---- optional Jacobian hook ----
        self.jacobian_hook_call(active_joints, q_hist[0])
        self.jacobian_hook_call(active_joints, q_hist[-1])

        # ---- success ----
        resp.trajectory = traj
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        ee_state = self.robot.get_current_state()
        # rospy.loginfo("Current end-effector/hand state= $s", ee_state)
        return GetMotionPlanResponse(motion_plan_response=resp)


if __name__ == "__main__":
    rospy.init_node("dgm_planner_node")
    DGMPlannerService()
    rospy.spin()
