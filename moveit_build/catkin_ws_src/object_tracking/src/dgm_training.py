#!/usr/bin/env python3
import os
import time
from typing import List
import jax
import numpy as np
import rospy

from dataclasses import dataclass

from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

from object_tracking.dgm_jax import (
    adam_init,
    init_mlp_params,
    make_batch,
    policy_grad_q_np,
    save_checkpoint,
    train_step,
)
from object_tracking.fk_client import FKClient
from datetime import datetime

from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from moveit_commander import RobotCommander, MoveGroupCommander

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes
from moveit_msgs.msg import RobotTrajectory, RobotState

from sensor_msgs.msg import JointState

from trajectory_msgs.msg import JointTrajectoryPoint

from dgm_planner_node import (
    validate_with_moveit_state_validity,
    robot_state_from_q,  # needed for single-state validity checks
)

"""
Reference
1. A. Al Aradi et al. (2018) Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning, pp 41-47
"""


def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


def default_vel_limits():
    # Conservative joint velocity limits (rad/s) for stable rollout
    return np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0], dtype=np.float64)


def finite(x: np.ndarray) -> bool:
    return np.all(np.isfinite(x))


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def sample_goals(n):
    xs = np.random.uniform(0.25, 0.65, (n, 1))
    ys = np.random.uniform(-0.30, 0.30, (n, 1))
    zs = np.random.uniform(0.10, 0.60, (n, 1))
    return np.hstack([xs, ys, zs]).astype(np.float64)


def position_loss_fn(fk, joint_names, batch, Qp, g_np, q_np):
    l_np = np.zeros((batch,), dtype=np.float64)
    for i in range(batch):
        try:
            p = fk.ee_position(joint_names, q_np[i])
            rospy.loginfo("p: %s", p)
            e = p - g_np[i]
            l_np[i] = Qp * float(np.dot(e, e))
        except Exception:
            rospy.logwarn("fk_pos: couldn't retrieve fk position")
            l_np[i] = 1e3
    return l_np


def dist_term(value, min_limit, max_limit):
    d = min(value - min_limit, max_limit - value)
    return d ** 2


def joint_limit_penalty(q_row, jmin, jmax, eps=1e-6):
    penalty = 0.0
    for k in range(len(q_row)):
        d2 = dist_term(q_row[k], jmin[k], jmax[k])
        penalty += 1.0 / (d2 + eps)
    return penalty


def batch_joint_limit_penalty(q_batch, jmin, jmax, eps=1e-6):
    out = np.zeros((q_batch.shape[0],), dtype=np.float64)
    for i in range(q_batch.shape[0]):
        out[i] = joint_limit_penalty(q_batch[i], jmin, jmax, eps=eps)
    return out


def is_state_valid(svc, active_joints, q_row, group_name):
    """
    Check a single joint configuration with MoveIt's /check_state_validity.
    """
    req = GetStateValidityRequest()
    req.group_name = group_name
    req.robot_state = robot_state_from_q(active_joints, q_row)
    try:
        resp = svc(req)
        return bool(resp.valid)
    except Exception as e:
        rospy.logwarn("State validity service call failed: %s", e)
        return False


def filter_valid_q_batch(svc, active_joints, q_batch, group_name):
    """
    Filter an existing batch of q states, keeping only MoveIt-valid states.
    """
    valid_rows = []
    for i in range(q_batch.shape[0]):
        if is_state_valid(svc, active_joints, q_batch[i], group_name):
            valid_rows.append(q_batch[i])

    if len(valid_rows) == 0:
        return np.empty((0, q_batch.shape[1]), dtype=np.float64)

    return np.asarray(valid_rows, dtype=np.float64)


@dataclass
class RolloutConfig:
    T: float = 2.0
    dt: float = 0.02
    vel_limits: np.ndarray = None  # (7,)
    joint_min: np.ndarray = None  # (7,)
    joint_max: np.ndarray = None  # (7,)
    R_diag: np.ndarray = None  # (7,)
    max_nan_guard: int = 5


def rollout_sampling_batch(
        batch,
        initial_q,
        batch_t,
        cfg,
        goal_pos,
        active_joints,
        model,
        device="cpu"):
    """
    For rollout training, we need to sample initial states and then do a forward rollout 
    using the current policy to get state-action pairs for training.

    The HJB residual loss only needs state samples, but the rollout loss needs state-action pairs, 
    so we need to compute the optimal control u* for each sampled state in order to do rollouts and get the next states.
    """

    rolls = range(batch)

    q = np.asarray(initial_q, dtype=np.float64).copy()

    traj = RobotTrajectory()
    traj.joint_trajectory.joint_names = list(active_joints)

    batch_samples = np.zeros((batch, 7), dtype=np.float64)
    nan_hits = 0

    # Precompute R^{-1}
    R_inv = 1.0 / np.maximum(cfg.R_diag, 1e-9)

    k = 0
    dt = batch_t[0]
    for k in rolls:

        t = float(np.asarray(batch_t[k]).reshape(-1)[0])
        t_norm = t / max(cfg.T, 1e-9)

        batch_samples[k, :] = q

        # Add trajectory point at current q (velocities set below)
        pt = JointTrajectoryPoint()
        pt.positions = q.tolist()
        pt.time_from_start = rospy.Duration.from_sec(t)

        grad_q_np = policy_grad_q_np(model, q, t_norm, goal_pos)

        if not np.all(np.isfinite(grad_q_np)):
            nan_hits += 1
            if nan_hits > cfg.max_nan_guard:
                raise RuntimeError("DGM rollout: too many non-finite gradients; aborting.")
            # fallback: zero velocity
            u = np.zeros(7, dtype=np.float64)
        else:
            # u* = -0.5 R^{-1} grad
            u = -0.5 * R_inv * grad_q_np

        # clamp velocities
        u = clamp(u, -cfg.vel_limits, cfg.vel_limits)

        pt.velocities = u.tolist()
        traj.joint_trajectory.points.append(pt)

        # integrate forward (except after last point)

        if k < len(batch_t) - 1:
            if k == 0:
                dt = float(np.asarray(batch_t[k]).reshape(-1)[0])
            else:
                del_t = float(np.asarray(batch_t[k]).reshape(-1)[0] - np.asarray(batch_t[k - 1]).reshape(-1)[0])
                if del_t > 0: # 
                    dt = max(
                        1e-3,
                        float(np.asarray(batch_t[k]).reshape(-1)[0] - np.asarray(batch_t[k - 1]).reshape(-1)[0]),
                    )
                # keep prev dt if del_t is non-positive, to avoid NaNs and instability in rollouts due to bad batch_t samples.
            q = q + dt * u
            q = clamp(q, cfg.joint_min, cfg.joint_max)

        k += 1

    # rospy.loginfo("Rollout sampling done: initial_q: %s, q_hist %s", initial_q.shape, batch_samples.shape)

    # rospy.loginfo("Rollout sampling done. nan_hits: %d / %d", nan_hits, rollout_length)

    return batch_samples


def sample_valid_q_batch(
        svc,
        active_joints,
        group_name,
        jmin,
        jmax,
        batch_size,
        max_attempt_factor=20,
        candidate_multiplier=2,
):
    """
    Rejection-sample joint states until we collect batch_size valid MoveIt states.

    Returns:
        q_valid: (batch_size, 7)
    Raises:
        RuntimeError if not enough valid states could be collected.
    """
    collected = []
    attempts = 0
    max_attempts = max_attempt_factor * batch_size

    while len(collected) < batch_size and attempts < max_attempts:
        n_needed = batch_size - len(collected)
        n_candidates = max(n_needed * candidate_multiplier, 8)

        q_candidates = np.random.uniform(jmin, jmax, (n_candidates, 7)).astype(np.float64)

        for i in range(q_candidates.shape[0]):
            attempts += 1
            if is_state_valid(svc, active_joints, q_candidates[i], group_name):
                collected.append(q_candidates[i])
                if len(collected) >= batch_size:
                    break
            if attempts >= max_attempts:
                break

    if len(collected) < batch_size:
        raise RuntimeError(
            f"Could only collect {len(collected)}/{batch_size} valid states "
            f"after {attempts} validity checks."
        )

    return np.asarray(collected, dtype=np.float64)


def main_():
    rospy.init_node("dgm_pretrain")

    print(f"jax version: {jax.__version__}")
    print(f"numpy version: {np.__version__}")
    rospy.loginfo("JAX devices: %s", jax.devices())

    device = rospy.get_param("~device", "cpu")
    T = float(rospy.get_param("~T", 2.0))

    iters = int(rospy.get_param("~iters", 3000))
    ft_iters = int(rospy.get_param("~ft_iters", 1000))
    batch = int(rospy.get_param("~batch", 192))
    lr = float(rospy.get_param("~lr", 3e-4))
    hidden = int(rospy.get_param("~hidden", 192))
    depth = int(rospy.get_param("~depth", 24))

    print(f"device: {device}")

    Qp = float(rospy.get_param("~Qp", 10.0))
    QpT = float(rospy.get_param("~Qp_terminal", 100.0))

    Cj = float(rospy.get_param("~Cj", 0.0))
    Cv = float(rospy.get_param("~Cv", 0.0))
    Ctr = float(rospy.get_param("~Ctr", 0.01))
    Cpd = float(rospy.get_param("~Cpd", 1000.0))

    R_diag = np.array(rospy.get_param("~R_diag", [0.15] * 7), dtype=np.float64)
    R_inv_diag = 1.0 / R_diag

    # vel_limits_np = np.array(
    #     rospy.get_param("~vel_limits", [2.0] * 7),
    #     dtype=np.float64
    # )

    group_name = rospy.get_param("~group_name", "panda_arm")

    group = MoveGroupCommander(group_name)
    joint_names = group.get_active_joints()
    if len(joint_names) != 7:
        raise RuntimeError(f"Expected 7 joints, got {len(joint_names)}: {joint_names}")

    fk = FKClient(service="/compute_fk", ee_link="panda_hand", frame="world")
    jmin, jmax = panda_joint_limits()

    # MoveIt state validity service
    state_validity_service = rospy.get_param("~state_validity_service", "/check_state_validity")
    rospy.loginfo("Waiting for state validity service: %s", state_validity_service)
    rospy.wait_for_service(state_validity_service)
    validity_svc = rospy.ServiceProxy(state_validity_service, GetStateValidity)

    seed = int(rospy.get_param("~seed", 0))
    model = init_mlp_params(jax.random.PRNGKey(seed), in_dim=11, hidden=hidden, depth=depth)
    opt_state = adam_init(model)

    t0 = time.time()
    loss = 0.0

    t_loss = 0.0
    t_loss_pde = 0.0
    t_loss_term = 0.0
    itr = 50

    now = str(datetime.now()).replace("-", "").replace(" ", ":")
    results_dir = rospy.get_param(
        "~results_dir",
        "/root/catkin_ws/src/object_tracking/results"
    )
    results_file = now + ".csv"
    data_header = f"time,t_loss_pde,t_loss_term,t_loss\n"
    results_preamble = (
        f"DGMValueNetJAX(in_dim=11, hidden={hidden}, depth={depth})\n"
        f"lr:{lr}\n"
        f"Cj:{Cj},Cv:{Cv},Ctr:{Ctr},Cpd:{Cpd}\n"
        f"{data_header}"
    )

    print(f"results_file {results_file}\n \n results_preamble {results_preamble}")

    out_path = rospy.get_param(
        "~out_path",
        "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pkl"
    )

    train_perf_data_path = rospy.get_param(
        "~train_perf_path",
        "/root/catkin_ws/src/object_tracking/models/train_perf_data.csv"
    )

    os.makedirs(os.path.dirname(train_perf_data_path), exist_ok=True)

    with open(train_perf_data_path, "a") as f:
        f.write(results_preamble)

    group_name = rospy.get_param("~group_name", "panda_arm")

    cfg = RolloutConfig(
        T=T,
        dt=0.02,
        vel_limits=default_vel_limits(),
        joint_min=panda_joint_limits()[0],
        joint_max=panda_joint_limits()[1],
        R_diag=R_diag,
        max_nan_guard=int(rospy.get_param("~max_nan_guard", 5)),
    )

    # PRE-TRAIN WITH REGULAR DOMAIN SAMPLING
    for it in range(1, iters + 1):

        # PDE / rollout training batch

        q_np = sample_valid_q_batch(
            svc=validity_svc,
            active_joints=joint_names,
            group_name=group_name,
            jmin=jmin,
            jmax=jmax,
            batch_size=batch,
        )

        # Optional audit using trajectory validator style.
        # Since q_np is a stack of states, this checks the whole batch as a "trajectory".
        ok, first_bad_idx, msg = validate_with_moveit_state_validity(
            svc=validity_svc,
            active_joints=joint_names,
            q_hist=q_np,
            group_name=group_name,
            stride=1,
        )
        if not ok:
            rospy.logwarn(
                "Unexpected invalid state after filtering at index %d: %s. Resampling batch.",
                first_bad_idx, msg
            )
            q_np = sample_valid_q_batch(
                svc=validity_svc,
                active_joints=joint_names,
                group_name=group_name,
                jmin=jmin,
                jmax=jmax,
                batch_size=batch,
            )

        t_np = np.random.uniform(0.0, T, (batch, 1)).astype(np.float64)
        t_np = np.sort(t_np.flatten()).reshape((batch, 1))
        g_np = sample_goals(batch)

        pos_cost_np = np.zeros((batch,), dtype=np.float64)
        for i in range(batch):
            try:
                p = fk.ee_position(joint_names, q_np[i])
                e = p - g_np[i]
                pos_cost_np[i] = 0.5 * Qp * float(np.dot(e, e))
            except Exception:
                rospy.logwarn("fk_pos l: couldn't retrieve fk position")
                pos_cost_np[i] = 1e3

        joint_limit_penalty_np = batch_joint_limit_penalty(q_np, jmin, jmax)
        l_np = pos_cost_np + Cj * joint_limit_penalty_np

        bt = max(128, batch // 3)

        # Terminal batch

        qT_np = sample_valid_q_batch(
            svc=validity_svc,
            active_joints=joint_names,
            group_name=group_name,
            jmin=jmin,
            jmax=jmax,
            batch_size=bt,
        )
        gT_np = sample_goals(bt)

        phi_np = np.zeros((bt,), dtype=np.float64)
        for i in range(bt):
            try:
                p = fk.ee_position(joint_names, qT_np[i])
                e = p - gT_np[i]
                phi_np[i] = 0.5 * QpT * float(np.dot(e, e))
            except Exception:
                rospy.logwarn("fk_pos phi: couldn't retrieve fk position")
                phi_np[i] = 1e3

        tT_np = np.ones((bt, 1), dtype=np.float64)

        # Sample a few rows from the TC to add to the PDE batch, to help with training stability.
        # This is a bit hacky but seems to help.
        n_samples = 8
        indices = np.random.permutation(bt)[:n_samples]
        l_np[-n_samples:] = phi_np[indices]

        train_batch = make_batch(
            q_np=q_np,
            t_np=t_np / T,
            g_np=g_np,
            running_cost_np=l_np,
            q_t_np=qT_np,
            t_t_np=tT_np,
            g_t_np=gT_np,
            phi_t_np=phi_np,
            r_inv_diag_np=R_inv_diag,
        )

        model, opt_state, loss, loss_pde, loss_term = train_step(model, opt_state, train_batch, lr, Cpd, Ctr)
        loss = float(loss)
        loss_pde = float(loss_pde)
        loss_term = float(loss_term)
        t_loss_pde += loss_pde
        t_loss_term += loss_term
        t_loss += loss

        if it % itr == 0:
            rospy.loginfo(
                "iter=%d  pde=%.6f term=%.6f loss=%.6f elapsed=%.1fs",
                it,
                float(t_loss_pde) / (itr * (batch + bt)),
                float(t_loss_term) / (itr * (batch + bt)),
                float(t_loss) / (itr * (batch + bt)),
                time.time() - t0
            )
            data_line = (f"{it}.2fs"
                         f"{float(t_loss_pde) / (itr * (batch + bt))}"
                         f",{float(t_loss_term) / (itr * (batch + bt))}"
                         f",{float(t_loss) / (itr * (batch + bt))}"
                         )
            with open(train_perf_data_path, "a") as f:
                f.write(data_line + "\n")

            t_loss_pde = 0.0
            t_loss_term = 0.0
            t_loss = 0.0

    f.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_checkpoint(
        out_path,
        model,
        opt_state,
        meta={"in_dim": 11, "hidden": hidden, "depth": depth, "T": T, "framework": "jax"},
        loss=float(loss),
    )

    rospy.loginfo("Saved epoch checkpoint: %s", out_path)
    rospy.loginfo("PRE-TRAIN DONE. Saved final: %s", out_path)

    # FINE TUNING WITH ROLLOUT SAMPLING
    for it in range(1, ft_iters + 1):

        # PDE / rollout training batch

        q_np = sample_valid_q_batch(
            svc=validity_svc,
            active_joints=joint_names,
            group_name=group_name,
            jmin=jmin,
            jmax=jmax,
            batch_size=batch,
        )

        # Optional audit using trajectory validator style.
        # Since q_np is a stack of states, this checks the whole batch as a "trajectory".
        ok, first_bad_idx, msg = validate_with_moveit_state_validity(
            svc=validity_svc,
            active_joints=joint_names,
            q_hist=q_np,
            group_name=group_name,
            stride=1,
        )
        if not ok:
            rospy.logwarn(
                "Unexpected invalid state after filtering at index %d: %s. Resampling batch.",
                first_bad_idx, msg
            )
            q_np = sample_valid_q_batch(
                svc=validity_svc,
                active_joints=joint_names,
                group_name=group_name,
                jmin=jmin,
                jmax=jmax,
                batch_size=batch,
            )

        t_np = np.random.uniform(0.0, T, (batch, 1)).astype(np.float64)
        t_np = np.sort(t_np.flatten()).reshape((batch, 1))
        g_np = sample_goals(batch)

        resp = MotionPlanResponse()
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        resp.planning_time = 0.0

        group = MoveGroupCommander(group_name)
        active_joints = group.get_active_joints()

        rollout_samples = rollout_sampling_batch(
            batch=batch,
            initial_q=q_np[0],
            batch_t=t_np,
            cfg=cfg,
            goal_pos=g_np[0],
            active_joints=active_joints,
            model=model)

        # rospy.loginfo("Rollout samples, batch shapes: %s, %s", rollout_samples.shape, q_np.shape)

        pos_cost_np = np.zeros((batch,), dtype=np.float64)
        for i in range(batch):
            try:
                p = fk.ee_position(joint_names, q_np[i])
                e = p - g_np[i]
                pos_cost_np[i] = 0.5 * Qp * float(np.dot(e, e))
            except Exception:
                rospy.logwarn("fk_pos l: couldn't retrieve fk position")
                pos_cost_np[i] = 1e3

        joint_limit_penalty_np = batch_joint_limit_penalty(q_np, jmin, jmax)
        l_np = pos_cost_np + Cj * joint_limit_penalty_np

        bt = max(64, batch // 3)

        # Terminal batch

        qT_np = sample_valid_q_batch(
            svc=validity_svc,
            active_joints=joint_names,
            group_name=group_name,
            jmin=jmin,
            jmax=jmax,
            batch_size=bt,
        )
        gT_np = sample_goals(bt)

        phi_np = np.zeros((bt,), dtype=np.float64)
        for i in range(bt):
            try:
                p = fk.ee_position(joint_names, qT_np[i])
                e = p - gT_np[i]
                phi_np[i] = 0.5 * QpT * float(np.dot(e, e))
            except Exception:
                rospy.logwarn("fk_pos phi: couldn't retrieve fk position")
                phi_np[i] = 1e3

        tT_norm_np = np.ones((bt, 1), dtype=np.float64)
        tT_abs_np = np.full((bt, 1), T, dtype=np.float64)
        rollout_samples_T = rollout_sampling_batch(
            batch=bt,
            initial_q=qT_np[0],
            batch_t=tT_abs_np,
            cfg=cfg,
            goal_pos=gT_np[0],
            active_joints=active_joints,
            model=model)

        # rospy.loginfo("Rollout T samples. gT shapes: %s, %s", rollout_samples_T.shape, qT_np.shape)

        # Sample a few rows from the TC to add to the PDE batch, to help with training stability.
        # This is a bit hacky but seems to help.
        n_samples = 8
        indices = np.random.permutation(bt)[:n_samples]
        l_np[-n_samples:] = phi_np[indices]

        train_batch = make_batch(
            q_np=rollout_samples,
            t_np=t_np / T,
            g_np=g_np,
            running_cost_np=l_np,
            q_t_np=rollout_samples_T,
            t_t_np=tT_norm_np,
            g_t_np=gT_np,
            phi_t_np=phi_np,
            r_inv_diag_np=R_inv_diag,
        )

        model, opt_state, loss, loss_pde, loss_term = train_step(model, opt_state, train_batch, lr, Cpd, Ctr)
        loss = float(loss)
        loss_pde = float(loss_pde)
        loss_term = float(loss_term)
        t_loss_pde += loss_pde
        t_loss_term += loss_term
        t_loss += loss

        if it % itr == 0:
            rospy.loginfo(
                "iter=%d  pde=%.6f term=%.6f loss=%.6f elapsed=%.1fs",
                it,
                float(t_loss_pde) / (itr * (batch + bt)),
                float(t_loss_term) / (itr * (batch + bt)),
                float(t_loss) / (itr * (batch + bt)),
                time.time() - t0
            )
            data_line = (f"{it}"
                         f",{float(t_loss_pde) / (itr * (batch + bt))}"
                         f",{float(t_loss_term) / (itr * (batch + bt))}"
                         f",{float(t_loss) / (itr * (batch + bt))}"
                         )
            with open(train_perf_data_path, "a") as f:
                f.write(data_line + "\n")

            t_loss_pde = 0.0
            t_loss_term = 0.0
            t_loss = 0.0

    f.close()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_checkpoint(
        out_path,
        model,
        opt_state,
        meta={"in_dim": 11, "hidden": hidden, "depth": depth, "T": T, "framework": "jax"},
        loss=float(loss),
    )

    rospy.loginfo("Saved epoch checkpoint: %s", out_path)
    rospy.loginfo("FINE-TUNE DONE. Saved final: %s", out_path)


if __name__ == "__main__":
    main_()
