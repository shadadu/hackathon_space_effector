#!/usr/bin/env python3
import os
import time
import numpy as np
import rospy
import torch
import torch.optim as optim

from moveit_commander import MoveGroupCommander
from object_tracking.dgm_model import DGMValueNet, ValueNet, ValueNet_, build_input, save_checkpoint
from object_tracking.hjb_loss import hjb_residual_loss, hjb_residual_loss_, terminal_loss
from object_tracking.fk_client import FKClient
from datetime import datetime

from .dgm_planner_node import validate_with_moveit_state_validity

"""
Reference 
1. A. Al Aradi et al. (2018) Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning, pp 41-47
"""


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def sample_goals(n):
    # Sample from xyz boundaries (cuboid) from which
    # to train the NN to generate plans to reach points inside this goals cuboid/region
    xs = np.random.uniform(0.25, 0.65, (n, 1))
    ys = np.random.uniform(-0.30, 0.30, (n, 1))
    zs = np.random.uniform(0.10, 0.60, (n, 1))
    return np.hstack([xs, ys, zs]).astype(np.float64)


def position_loss_fn(fk, joint_names, batch, Qp, g_np, q_np):
    l_np = np.zeros((batch,), dtype=np.float64)
    # rospy.loginfo("l shape: %s", l_np.shape)
    for i in range(batch):
        try:
            p = fk.ee_position(joint_names, q_np[i])  # fk client gets coordinate position of hand/end-effector
            rospy.loginfo("p: %s", p)
            e = p - g_np[i]  # distance between current joint position i and goal position i
            l_np[i] = Qp * float(np.dot(e, e))
        except Exception:
            rospy.logwarn("fk_pos: couldn't retrieve fk position")
            l_np[i] = 1e3
    # rospy.loginfo("Loss vector %s", l_np)
    return l_np


def dist_term(value, min_limit, max_limit):
    """
    Squared distance to the nearest admissible boundary.
    Small near the middle, goes to 0 at the boundary.
    We convert this into a penalty by inverting it in the helpers below.
    """
    d = min(value - min_limit, max_limit - value)
    return d ** 2


def joint_limit_penalty(q_row, jmin, jmax, eps=1e-6):
    """
    Large penalty when q is near either joint limit.
    Uses inverse of dist_term so penalty grows near the boundary.
    """
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

def main_():
    rospy.init_node("dgm_pretrain")

    print(f"torch version: {torch.__version__}")
    print(f"numpy version: {np.__version__}")

    device = rospy.get_param("~device", "cpu")
    T = float(rospy.get_param("~T", 2.0))

    iters = int(rospy.get_param("~iters", 3000))
    batch = int(rospy.get_param("~batch", 192))
    lr = float(rospy.get_param("~lr", 3e-4))
    hidden = int(rospy.get_param("~hidden", 256))
    depth = int(rospy.get_param("~depth", 4))

    print(f"device: {device}")

    Qp = float(rospy.get_param("~Qp", 10.0))
    QpT = float(rospy.get_param("~Qp_terminal", 80.0))

    # New weights for limit penalties
    Cj = float(rospy.get_param("~Cj", 0.01))  # joint position limit penalty weight
    Cv = float(rospy.get_param("~Cv", 0.001))  # joint velocity limit penalty weight
    Ctr = float(rospy.get_param("~Ctr", 100.0))  # joint velocity limit penalty weight
    Cpd = float(rospy.get_param("~Cpd", 0.01))

    # State/control weighting
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

    group = MoveGroupCommander("panda_arm")
    joint_names = group.get_active_joints()
    if len(joint_names) != 7:
        raise RuntimeError(f"Expected 7 joints, got {len(joint_names)}: {joint_names}")

    fk = FKClient(service="/compute_fk", ee_link="panda_hand", frame="world")
    jmin, jmax = panda_joint_limits()

    # model = DGMValueNet(in_dim=11, hidden=hidden, depth=depth).to(device)
    model = ValueNet(num_layers=8, input_dim=11, output_dim=1, hidden_size=192)
    # model = ValueNet_(num_layers=8, input_dim=11, output_dim=1, hidden_size=192, expansion_factor=1)

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
    results_preamble = str(model) + "\n" + \
                       f"lr:{lr}" + "\n" \
                                    f"Cj:{Cj},Cv:{Cv},Ctr:{Ctr},Cpd:{Cpd}"

    print(f"results_file {results_file}\n \n results_preamble {results_preamble}")

    for it in range(1, iters + 1):
        q_np = np.random.uniform(jmin, jmax, (batch, 7)).astype(np.float64)
        t_np = np.random.uniform(0.0, T, (batch, 1)).astype(np.float64)
        g_np = sample_goals(batch)

        # Position running cost via FK
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

        # Running cost includes:
        # l = position_cost + Cj * joint_limit_penalty + Cv * joint_velocity_limit_penalty
        #
        # The joint_velocity_limit_penalty depends on the optimal control u*(q,t,g),
        # so that part is added inside hjb_residual_loss(...).
        l_np = pos_cost_np + Cj * joint_limit_penalty_np

        q = torch.tensor(q_np, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor(t_np / T, dtype=torch.float32, device=device, requires_grad=True)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)
        l = torch.tensor(l_np, dtype=torch.float32, device=device)

        # V = model(build_input(q, t, g))
        V = model(build_input(q, t, g), build_input(q, t, g))
        # rospy.loginfo("Shape of residual loss inputs %s, %s, %s", V.shape, q.shape, t.shape)

        loss_pde = hjb_residual_loss(V, q, t, l,
                                     R_inv_diag)  # hjb_residual_loss(V, q, t_norm, running_cost, R_inv_diag)

        # loss_pde = hjb_residual_loss_(
        #     V=V,
        #     q=q,
        #     t=t,
        #     running_cost=l,
        #     R_inv_diag=R_inv_diag.to(device),
        #     vel_limits=vel_limits,
        #     Cv=Cv
        # )

        # Terminal batch
        bt = max(64, batch // 3)
        qT_np = np.random.uniform(jmin, jmax, (bt, 7)).astype(np.float64)
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
        loss_term = terminal_loss(VT, phi)

        loss = Cpd * loss_pde + Ctr * loss_term
        t_loss_pde += loss_pde.item()
        t_loss_term += loss_term.item()
        t_loss += loss.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if it % itr == 0:
            rospy.loginfo(
                "iter=%d  pde=%.6f term=%.6f loss=%.6f elapsed=%.1fs ",
                it,
                float(t_loss_pde) / (itr * (batch + bt)),
                float(t_loss_term) / (itr * (batch + bt)),
                float(t_loss) / (itr * (batch + bt)),
                time.time() - t0
            )
            t_loss = 0.0
            t_loss_pde = 0.0
            t_loss_term = 0.0

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


if __name__ == "__main__":
    main_()
