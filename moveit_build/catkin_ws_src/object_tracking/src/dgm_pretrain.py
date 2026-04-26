#!/usr/bin/env python3
import os
import time
import numpy as np
import rospy
import torch
import torch.optim as optim

from moveit_commander import MoveGroupCommander
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

from object_tracking.dgm_model import DGMValueNet, ValueNet, ValueNet_, build_input, save_checkpoint
from object_tracking.hjb_loss import hjb_residual_loss, hjb_residual_loss_, terminal_loss
from object_tracking.fk_client import FKClient
from datetime import datetime

from dgm_planner_node import (
    validate_with_moveit_state_validity,
    robot_state_from_q,  # needed for single-state validity checks
)

"""
Reference
1. A. Al Aradi et al. (2018) Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning, pp 41-47
"""


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

    print(f"torch version: {torch.__version__}")
    print(f"numpy version: {np.__version__}")

    device = rospy.get_param("~device", "cpu")
    T = float(rospy.get_param("~T", 10.0))

    iters = int(rospy.get_param("~iters", 3000))
    batch = int(rospy.get_param("~batch", 192))
    lr = float(rospy.get_param("~lr", 3e-4))
    hidden = int(rospy.get_param("~hidden", 256))
    depth = int(rospy.get_param("~depth", 4))

    print(f"device: {device}")

    Qp = float(rospy.get_param("~Qp", 10.0))
    QpT = float(rospy.get_param("~Qp_terminal", 80.0))

    Cj = float(rospy.get_param("~Cj", 0.001))
    Cv = float(rospy.get_param("~Cv", 0.001))
    Ctr = float(rospy.get_param("~Ctr", 1000.0))

    R_diag = torch.tensor(rospy.get_param("~R_diag", [0.15] * 7), dtype=torch.float32)
    print(f"R_diag {R_diag}")
    R_inv_diag = 1.0 / R_diag

    vel_limits_np = np.array(
        rospy.get_param("~vel_limits", [2.0] * 7),
        dtype=np.float64
    )
    vel_limits = torch.tensor(vel_limits_np, dtype=torch.float32, device=device)

    out_path = rospy.get_param(
        "~out_path",
        "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pth"
    )

    train_perf_data_path = rospy.get_param(
        "~train_perf_path",
        "/root/catkin_ws/src/object_tracking/models/train_perf_data.csv"
    )

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

    # model = DGMValueNet(in_dim=11, hidden=hidden, depth=depth).to(device)
    model = ValueNet(num_layers=8, input_dim=11, output_dim=1, hidden_size=192)
    # model = ValueNet_(num_layers=12, input_dim=11, output_dim=1, hidden_size=192, expansion_factor=2)

    opt = optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    loss = torch.tensor(0.0, device=device)

    t_loss = 0.0
    t_loss_pde = 0.0
    t_loss_term = 0.0
    itr = 10

    now = str(datetime.now()).replace("-", "").replace(" ", ":")
    results_dir = "/Users/rckyi/Documents/GitHub/hackathon_space_effector/moveit_build/"
    results_file = now + ".csv"
    data_header = f"time,t_loss_pde,t_loss_term,t_loss\n"
    results_preamble = str(model) + "\n" + \
                       f"lr:{lr}" + "\n" \
                                    f"Cj:{Cj},Cv:{Cv},Ctr:{Ctr}\n" + data_header

    print(f"results_file {results_file}\n \n results_preamble {results_preamble}")

    total_validity_checks = 0
    total_rejected_states = 0

    os.makedirs(os.path.dirname(train_perf_data_path), exist_ok=True)

    with open(train_perf_data_path, "a") as f:
        f.write(results_preamble)

    # with open(train_perf_data_path, "w") as f:
    #     f.write(results_preamble + "\n\n")

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
        g_np = sample_goals(batch)

        pos_cost_np = np.zeros((batch,), dtype=np.float64)
        for i in range(batch):
            try:
                p = fk.ee_position(joint_names, q_np[i])
                e = p - g_np[i]
                pos_cost_np[i] = Qp * float(np.dot(e, e))
            except Exception:
                rospy.logwarn("fk_pos l: couldn't retrieve fk position")
                pos_cost_np[i] = 1e3

        joint_limit_penalty_np = batch_joint_limit_penalty(q_np, jmin, jmax)
        l_np = pos_cost_np + Cj * joint_limit_penalty_np

        q = torch.tensor(q_np, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor(t_np / T, dtype=torch.float32, device=device, requires_grad=True)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)
        l = torch.tensor(l_np, dtype=torch.float32, device=device)

        V = model(build_input(q, t, g), build_input(q, t, g))
        # V = model(build_input(q, t, g))

        # loss_pde = hjb_residual_loss(
        #     V,
        #     q,
        #     t,
        #     l,
        #     R_inv_diag
        # )

        #
        loss_pde = hjb_residual_loss_(
            V=V,
            q=q,
            t=t,
            running_cost=l,
            R_inv_diag=R_inv_diag.to(device),
            vel_limits=vel_limits,
            Cv=Cv
        )

        # Terminal batch

        # bt = max(64, batch // 3)
        bt = max(16, batch // 3)

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
                phi_np[i] = QpT * float(np.dot(e, e))
            except Exception:
                rospy.logwarn("fk_pos phi: couldn't retrieve fk position")
                phi_np[i] = 1e3

        qT = torch.tensor(qT_np, dtype=torch.float32, device=device)
        tT = torch.ones((bt, 1), dtype=torch.float32, device=device, requires_grad=True)
        gT = torch.tensor(gT_np, dtype=torch.float32, device=device)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=device)

        VT = model(build_input(qT, tT, gT), build_input(qT, tT, gT))
        # VT = model(build_input(qT, tT, gT))

        loss_term = terminal_loss(VT, phi)
        loss = loss_pde + Ctr * loss_term
        t_loss_pde += loss_pde.item()
        t_loss_term += loss_term.item()
        t_loss += loss.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

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

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "loss": float(loss.item()),
    }
    torch.save(checkpoint, out_path)

    rospy.loginfo("Saved epoch checkpoint: %s", out_path)
    rospy.loginfo("DONE. Saved final: %s", out_path)
    rospy.loginfo("Total training time = %s mins", str((time.time() - t0) / 60))
    f.close()


if __name__ == "__main__":
    main_()
