#!/usr/bin/env python3
import os
import time
import threading
import queue
import numpy as np
import rospy
import torch
import torch.optim as optim

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes
from moveit_commander import MoveGroupCommander

from object_tracking.dgm_model import load_checkpoint, DGMValueNet
from object_tracking.dgm_rollout import rollout_value_policy
from object_tracking.hjb_loss import terminal_loss
from object_tracking.fk_client import FKClient


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float64)
    return jmin, jmax


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
        self.R_diag = np.array(rospy.get_param("~R_diag", [0.15]*7), dtype=np.float64)
        self.vel_limits = np.array(rospy.get_param("~vel_limits", [1.5,1.5,1.5,1.8,1.8,2.0,2.0]), dtype=np.float64)

        self.jmin, self.jmax = panda_joint_limits()

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
            rospy.loginfo("Online fine-tune ENABLED (lr=%g steps=%d budget=%gs)", self.ft_lr, self.ft_steps, self.ft_budget)

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
        mpr = req.motion_plan_request

        resp = MotionPlanResponse()
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        resp.planning_time = 0.0

        if not mpr.goal_constraints:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        with self.model_lock:
            model = self.model_ref["model"]
        if model is None:
            resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
            return GetMotionPlanResponse(motion_plan_response=resp)

        goal_pos = self.extract_goal_pos(mpr)
        q0 = self.start_q0(mpr)
        q0 = np.minimum(np.maximum(q0, self.jmin), self.jmax)

        # Queue for online fine-tune (non-blocking)
        if self.enable_finetune and self.finetuner is not None:
            self.finetuner.update_buffer(q0, goal_pos)

        t0 = time.time()
        with self.model_lock:
            traj = rollout_value_policy(
                model=self.model_ref["model"],
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
        resp.planning_time = float(time.time() - t0)
        resp.trajectory = traj
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        return GetMotionPlanResponse(motion_plan_response=resp)


if __name__ == "__main__":
    rospy.init_node("dgm_planner_node")
    DGMPlannerService()
    rospy.spin()
