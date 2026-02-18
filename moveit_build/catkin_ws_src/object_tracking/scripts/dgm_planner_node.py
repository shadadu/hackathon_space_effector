#!/usr/bin/env python3
import os
import time
import threading
import queue
import numpy as np
import rospy
import torch
import torch.optim as optim

from object_tracking.dgm_model import load_checkpoint, DGMValueNet
from object_tracking.dgm_rollout import rollout_value_policy
from object_tracking.hjb_loss import terminal_loss
from object_tracking.fk_client import FKClient

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes, RobotTrajectory

from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from trajectory_msgs.msg import JointTrajectoryPoint

# If you already have MoveGroupCommander elsewhere in the file:
from moveit_commander import MoveGroupCommander, RobotCommander

import time
import rospy




def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def _active_joint_positions_from_robot_state(robot_state, active_joints):
    """
    Extract q (len=ndof) in active_joints order from a moveit_msgs/RobotState.
    Returns None if any joint is missing.
    """
    if robot_state is None or robot_state.joint_state is None:
        return None
    names = list(robot_state.joint_state.name)
    pos = list(robot_state.joint_state.position)
    if not names or not pos or len(names) != len(pos):
        return None
    m = dict(zip(names, pos))
    try:
        return np.array([m[j] for j in active_joints], dtype=np.float64)
    except KeyError:
        return None


def _extract_goal_pose_from_motion_plan_request(mpr, default_frame="world"):
    """
    Extract a PoseStamped-like (frame_id, pose) from MotionPlanRequest goal_constraints.
    We assume your benchmark/intercept builder uses a PositionConstraint with primitive_poses[0].
    """
    if not mpr.goal_constraints:
        return None, None

    try:
        gc = mpr.goal_constraints[0]
        pc = gc.position_constraints[0]
        pose = pc.constraint_region.primitive_poses[0]
        frame_id = pc.header.frame_id if pc.header.frame_id else default_frame
        return frame_id, pose
    except Exception:
        return None, None


def interpolate_to_goal_via_ik(
        *,
        mpr,
        active_joints,
        ee_link,
        world_frame,
        ik_proxy,
        n_points=60,
        duration=2.0,
        ik_timeout_s=0.2,
        vel_limits=None,
        joint_min=None,
        joint_max=None,
):
    """
    Fallback planner:
      1) Extract goal pose from MotionPlanRequest constraints
      2) Call /compute_ik to get a joint solution
      3) Linearly interpolate q0 -> q1 over n_points (time_from_start over 'duration')
    Returns: moveit_msgs/RobotTrajectory OR None on failure.
    """
    goal_frame, goal_pose = _extract_goal_pose_from_motion_plan_request(mpr, default_frame=world_frame)
    if goal_pose is None:
        return None

    # Start state q0
    q0 = None
    if mpr.start_state and mpr.start_state.joint_state and mpr.start_state.joint_state.name:
        q0 = _active_joint_positions_from_robot_state(mpr.start_state, active_joints)

    # If no usable q0, we cannot safely interpolate (we don't assume current state in headless)
    if q0 is None:
        return None

    # Optional clamp start to limits
    if joint_min is not None and joint_max is not None:
        q0 = np.minimum(np.maximum(q0, joint_min), joint_max)

    # Build IK request
    ikreq = GetPositionIKRequest()
    ikreq.ik_request.group_name = mpr.group_name  # typically "panda_arm"
    ikreq.ik_request.ik_link_name = ee_link
    ikreq.ik_request.pose_stamped.header.frame_id = goal_frame
    ikreq.ik_request.pose_stamped.header.stamp = rospy.Time.now()
    ikreq.ik_request.pose_stamped.pose = goal_pose
    ikreq.ik_request.robot_state = mpr.start_state
    ikreq.ik_request.timeout = rospy.Duration.from_sec(float(ik_timeout_s))

    try:
        ikresp = ik_proxy(ikreq)
    except rospy.ServiceException:
        return None

    if ikresp.error_code.val != MoveItErrorCodes.SUCCESS:
        return None

    q1 = _active_joint_positions_from_robot_state(ikresp.solution, active_joints)
    if q1 is None:
        return None

    if joint_min is not None and joint_max is not None:
        q1 = np.minimum(np.maximum(q1, joint_min), joint_max)

    # Interpolate
    n = int(max(2, n_points))
    dur = float(max(1e-3, duration))

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    # Optional: crude time scaling by joint velocity limits (keeps it from being wildly infeasible)
    if vel_limits is not None:
        vel_limits = np.array(vel_limits, dtype=np.float64)
        # compute required duration so max |dq|/dt <= vel_limits
        dq = np.abs(q1 - q0)
        req_dur = float(np.max(dq / np.maximum(vel_limits, 1e-6)))
        # add a small cushion
        dur = max(dur, 1.2 * req_dur)

    for i in range(n):
        a = float(i) / float(n - 1)
        q = (1.0 - a) * q0 + a * q1
        t = a * dur

        p = JointTrajectoryPoint()
        p.positions = q.tolist()
        p.time_from_start = rospy.Duration.from_sec(t)
        traj.joint_trajectory.points.append(p)

    return traj


class OnlineFineTuner(threading.Thread):
    """
    Background fine-tuner: does small terminal-loss updates around recent (q, goal) samples.
    This is safe and fast, and sets up the pipeline for full HJB fine-tune next.
    """

    def __init__(self, model_ref, model_lock, joint_names, fk: FKClient, device="cpu",
                 lr=1e-4, steps_per_wake=10, time_budget_s=0.05, QpT=80.0):
        super().__init__(daemon=True)
        self.model_ref = model_ref
        self.model_lock = model_lock
        self.joint_names = joint_names
        self.fk = fk
        self.device = device
        self.lr = lr
        self.steps_per_wake = steps_per_wake
        self.time_budget_s = time_budget_s
        self.QpT = QpT

        self.buf = []  # list of (q0(7,), goal_pos(3,))
        self.buf_lock = threading.Lock()
        self.wake = threading.Event()
        self.stop_flag = threading.Event()

        self.opt = None

    def update_buffer(self, q0, goal_pos, maxlen=128):
        with self.buf_lock:
            self.buf.append((np.array(q0, dtype=np.float64), np.array(goal_pos, dtype=np.float64)))
            if len(self.buf) > maxlen:
                self.buf = self.buf[-maxlen:]
        self.wake.set()

    def run(self):
        while not self.stop_flag.is_set():
            self.wake.wait(timeout=0.5)
            self.wake.clear()
            if self.stop_flag.is_set():
                break

            # Snapshot buffer
            with self.buf_lock:
                if not self.buf:
                    continue
                batch = self.buf[-32:]  # small batch

            t_start = time.time()

            # Grab model
            with self.model_lock:
                model = self.model_ref["model"]
                if model is None:
                    continue
                model.train()
                if self.opt is None:
                    self.opt = optim.Adam(model.parameters(), lr=self.lr)

            # Train a few steps within time budget
            steps = 0
            while steps < self.steps_per_wake and (time.time() - t_start) < self.time_budget_s:
                # Random minibatch
                idx = np.random.choice(len(batch), size=min(16, len(batch)), replace=False)
                q_np = np.stack([batch[i][0] for i in idx], axis=0)
                g_np = np.stack([batch[i][1] for i in idx], axis=0)

                # Terminal targets via FK
                phi = []
                for i in range(q_np.shape[0]):
                    try:
                        p = self.fk.ee_position(self.joint_names, q_np[i])
                        e = p - g_np[i]
                        phi.append(self.QpT * float(np.dot(e, e)))
                    except Exception:
                        phi.append(1e3)
                phi = torch.tensor(phi, dtype=torch.float32, device=self.device)

                q = torch.tensor(q_np, dtype=torch.float32, device=self.device)
                tT = torch.ones((q.shape[0], 1), dtype=torch.float32, device=self.device)  # t=1
                g = torch.tensor(g_np, dtype=torch.float32, device=self.device)

                x = torch.cat([q, tT, g], dim=-1)
                with self.model_lock:
                    V = self.model_ref["model"](x)

                loss = terminal_loss(V, phi)

                with self.model_lock:
                    self.opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_ref["model"].parameters(), 5.0)
                    self.opt.step()

                steps += 1

            with self.model_lock:
                self.model_ref["model"].eval()


class DGMPlannerService:
    def __init__(self):

        self.service_name = rospy.get_param("~service_name", "/dgm/get_motion_plan")
        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.world_frame = rospy.get_param("~world_frame", "world")

        # rollout params
        self.T = float(rospy.get_param("~T", 2.0))
        self.dt = float(rospy.get_param("~dt", 0.02))
        self.R_diag = np.array(rospy.get_param("~R_diag", [0.15] * 7), dtype=np.float64)
        self.vel_limits = np.array(rospy.get_param("~vel_limits", [1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0]),
                                   dtype=np.float64)

        self.jmin, self.jmax = panda_joint_limits()

        self.mode = rospy.get_param("~mode", "dgm")  # "dgm" or "interpolate"
        self.n_points = int(rospy.get_param("~n_points", 60))
        self.duration = float(rospy.get_param("~duration", self.T))
        self.fallback = rospy.get_param("~fallback", "interpolate")  # if model missing: interpolate|fail

        # model
        self.device = rospy.get_param("~device", "cpu")
        self.model_path = rospy.get_param("~model_path", "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pt")

        self.model_ref = {"model": None, "meta": {}}
        self.model_lock = threading.Lock()

        # Load model if present
        if os.path.exists(self.model_path):
            model, meta = load_checkpoint(self.model_path, device=self.device)
            with self.model_lock:
                self.model_ref["model"] = model
                self.model_ref["meta"] = meta
            rospy.loginfo("Loaded DGM checkpoint: %s", self.model_path)
        else:
            rospy.logwarn("No checkpoint at %s (run rosrun object_tracking dgm_pretrain.py).", self.model_path)

        # Active joints
        group = MoveGroupCommander(self.group_name)
        self.active_joints = group.get_active_joints()
        if len(self.active_joints) != 7:
            raise RuntimeError(f"Expected 7 active joints for {self.group_name}, got {len(self.active_joints)}")

        # FK client for online fine-tune
        self.fk = FKClient(service="/compute_fk", ee_link=self.ee_link, frame=self.world_frame)

        # Online fine-tune config
        self.enable_finetune = bool(rospy.get_param("~enable_finetune", True))
        self.ft_lr = float(rospy.get_param("~finetune_lr", 1e-4))
        self.ft_steps = int(rospy.get_param("~finetune_steps_per_wake", 10))
        self.ft_budget = float(rospy.get_param("~finetune_time_budget_s", 0.05))
        self.QpT = float(rospy.get_param("~Qp_terminal", 80.0))

        self.finetuner = None
        if self.enable_finetune:
            self.finetuner = OnlineFineTuner(
                model_ref=self.model_ref,
                model_lock=self.model_lock,
                joint_names=self.active_joints,
                fk=self.fk,
                device=self.device,
                lr=self.ft_lr,
                steps_per_wake=self.ft_steps,
                time_budget_s=self.ft_budget,
                QpT=self.QpT,
            )
            self.finetuner.start()
            rospy.loginfo("Online fine-tune ENABLED (lr=%g steps=%d budget=%gs)", self.ft_lr, self.ft_steps,
                          self.ft_budget)

        self.srv = rospy.Service(self.service_name, GetMotionPlan, self.handle)
        rospy.loginfo("DGM service ready: %s", self.service_name)

    def extract_goal_pos(self, mpr):
        pc = mpr.goal_constraints[0].position_constraints[0]
        pose = pc.constraint_region.primitive_poses[0]
        return np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

    def start_q0(self, mpr):
        if mpr.start_state and mpr.start_state.joint_state.name:
            s_map = dict(zip(mpr.start_state.joint_state.name, mpr.start_state.joint_state.position))
            return np.array([s_map[j] for j in self.active_joints], dtype=np.float64)
        # fallback: mid-range
        return np.zeros(7, dtype=np.float64)

    def handle(self, req):
        """
        Robust service handler:
          - Never deadlocks on model_lock (copies model pointer, releases lock)
          - Supports:
              ~mode: "dgm" | "interpolate"
              ~fallback: "interpolate" | "fail"   (when model missing)
          - Has explicit time budgets so it won't hang your intercept loop
        """
        mpr = req.motion_plan_request

        resp = MotionPlanResponse()
        resp.error_code = MoveItErrorCodes()
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        resp.planning_time = 0.0

        # --- Basic validation ---
        if not mpr.goal_constraints:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Mode / fallback knobs
        mode = rospy.get_param("~mode", "dgm")  # "dgm" | "interpolate"
        fallback = rospy.get_param("~fallback", "interpolate")  # "interpolate" | "fail"

        # Timeouts
        max_handle_time_s = float(rospy.get_param("~max_handle_time_s", 0.35))  # keep intercept loop responsive
        ik_timeout_s = float(rospy.get_param("~ik_timeout_s", 0.20))

        # Interpolate params
        n_points = int(rospy.get_param("~n_points", 60))
        duration = float(rospy.get_param("~duration", getattr(self, "T", 2.0)))

        t_start = time.time()

        # Extract goal position for DGM rollout (your existing helpers)
        # (If your DGM only needs position-only, extract_goal_pos should return np.array shape (3,))
        try:
            goal_pos = self.extract_goal_pos(mpr)
        except Exception:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Start state q0 for DGM rollout
        try:
            q0 = self.start_q0(mpr)
            q0 = np.asarray(q0, dtype=np.float64)
        except Exception:
            resp.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Clamp to joint limits if available
        if hasattr(self, "jmin") and hasattr(self, "jmax"):
            q0 = np.minimum(np.maximum(q0, self.jmin), self.jmax)

        # Grab model pointer without holding lock during compute (avoids deadlocks + long lock holds)
        with self.model_lock:
            model = self.model_ref.get("model", None)

        # If user asked interpolate explicitly, do that now (even if model exists)
        if mode == "interpolate":
            # Ensure IK proxy exists
            if not hasattr(self, "ik"):
                # Lazy-create IK proxy from param (~ik_service), but don't crash if missing
                ik_service = rospy.get_param("~ik_service", "/compute_ik")
                try:
                    rospy.wait_for_service(ik_service, timeout=3.0)
                    self.ik = rospy.ServiceProxy(ik_service, GetPositionIK)
                except Exception:
                    resp.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                    return GetMotionPlanResponse(motion_plan_response=resp)

            traj = interpolate_to_goal_via_ik(
                mpr=mpr,
                active_joints=self.active_joints,
                ee_link=self.ee_link,
                world_frame=self.world_frame,
                ik_proxy=self.ik,
                n_points=n_points,
                duration=duration,
                ik_timeout_s=ik_timeout_s,
                vel_limits=getattr(self, "vel_limits", None),
                joint_min=getattr(self, "jmin", None),
                joint_max=getattr(self, "jmax", None),
            )
            resp.planning_time = float(time.time() - t_start)
            if traj is None:
                resp.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                return GetMotionPlanResponse(motion_plan_response=resp)
            resp.trajectory = traj
            resp.error_code.val = MoveItErrorCodes.SUCCESS
            return GetMotionPlanResponse(motion_plan_response=resp)

        # DGM requested, but model missing
        if model is None:
            if fallback == "interpolate":
                if not hasattr(self, "ik"):
                    ik_service = rospy.get_param("~ik_service", "/compute_ik")
                    try:
                        rospy.wait_for_service(ik_service, timeout=3.0)
                        self.ik = rospy.ServiceProxy(ik_service, GetPositionIK)
                    except Exception:
                        resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
                        return GetMotionPlanResponse(motion_plan_response=resp)

                traj = interpolate_to_goal_via_ik(
                    mpr=mpr,
                    active_joints=self.active_joints,
                    ee_link=self.ee_link,
                    world_frame=self.world_frame,
                    ik_proxy=self.ik,
                    n_points=n_points,
                    duration=duration,
                    ik_timeout_s=ik_timeout_s,
                    vel_limits=getattr(self, "vel_limits", None),
                    joint_min=getattr(self, "jmin", None),
                    joint_max=getattr(self, "jmax", None),
                )
                resp.planning_time = float(time.time() - t_start)
                if traj is None:
                    resp.error_code.val = MoveItErrorCodes.NO_IK_SOLUTION
                    return GetMotionPlanResponse(motion_plan_response=resp)
                resp.trajectory = traj
                resp.error_code.val = MoveItErrorCodes.SUCCESS
                return GetMotionPlanResponse(motion_plan_response=resp)

            resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Queue for online fine-tune (non-blocking)
        if getattr(self, "enable_finetune", False) and getattr(self, "finetuner", None) is not None:
            try:
                self.finetuner.update_buffer(q0, goal_pos)
            except Exception:
                pass

        # DGM rollout with time budget guard
        try:
            # If your rollout_value_policy is CPU-heavy, it can still block;
            # enforce a coarse time budget check before/after.
            if (time.time() - t_start) > max_handle_time_s:
                resp.error_code.val = MoveItErrorCodes.TIMED_OUT
                resp.planning_time = float(time.time() - t_start)
                return GetMotionPlanResponse(motion_plan_response=resp)

            traj = rollout_value_policy(
                model=model,  # use local pointer
                q0=q0,
                goal_pos=goal_pos,
                active_joints=self.active_joints,
                T=self.T,
                dt=self.dt,
                R_diag=self.R_diag,
                vel_limits=self.vel_limits,
                joint_min=self.jmin,
                joint_max=self.jmax,
                device=self.device,
            )

            resp.trajectory = traj
            resp.error_code.val = MoveItErrorCodes.SUCCESS
            resp.planning_time = float(time.time() - t_start)
            return GetMotionPlanResponse(motion_plan_response=resp)

        except Exception:
            resp.error_code.val = MoveItErrorCodes.PLANNING_FAILED
            resp.planning_time = float(time.time() - t_start)
            return GetMotionPlanResponse(motion_plan_response=resp)


if __name__ == "__main__":
    rospy.init_node("dgm_planner_node")
    DGMPlannerService()
    rospy.spin()
