#!/usr/bin/env python3
import os
import time
from datetime import datetime

import numpy as np
import rospy
import torch
import torch.optim as optim

from moveit_commander import MoveGroupCommander
from moveit_msgs.srv import GetStateValidity, GetStateValidityRequest

from object_tracking.dgm_model import ValueNet, ValueNet_, DGMValueNet, build_input, save_checkpoint
from object_tracking.hjb_loss import terminal_loss, hjb_residual, hjb_residual_
from object_tracking.fk_client import FKClient

from dgm_planner_node import (
    validate_with_moveit_state_validity,
    robot_state_from_q,
)


"""
DGM / HJB pretraining for Panda arm.
1. Samples only MoveIt-valid states.
2. Uses GLF-style global-local adaptive sampling. See Global-Local Fusion paper, Jiaqi Luoa et al.
3. Updates local sampling buffer using per-sample HJB residuals.
4. Keeps global sampling so the model does not collapse onto only hard regions.
"""


# ------------------------------------------------------------
# Panda limits and basic sampling
# ------------------------------------------------------------

def panda_joint_limits():
    jmin = np.array(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
        dtype=np.float64,
    )
    jmax = np.array(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
        dtype=np.float64,
    )
    return jmin, jmax


def sample_goals(n):
    """
    Sample target xyz goals inside the desired workspace cuboid.
    """
    xs = np.random.uniform(0.25, 0.65, (n, 1))
    ys = np.random.uniform(-0.30, 0.30, (n, 1))
    zs = np.random.uniform(0.10, 0.60, (n, 1))
    return np.hstack([xs, ys, zs]).astype(np.float64)


def clip_goals(g):
    """
    Clip local goal perturbations back into the training workspace.
    """
    g = np.asarray(g, dtype=np.float64)
    g[..., 0] = np.clip(g[..., 0], 0.25, 0.65)
    g[..., 1] = np.clip(g[..., 1], -0.30, 0.30)
    g[..., 2] = np.clip(g[..., 2], 0.10, 0.60)
    return g


# ------------------------------------------------------------
# Running-cost helpers
# ------------------------------------------------------------

def dist_term(value, min_limit, max_limit):
    """
    Squared distance to nearest joint-limit boundary.
    """
    d = min(value - min_limit, max_limit - value)
    return d ** 2


def joint_limit_penalty(q_row, jmin, jmax, eps=1e-6):
    """
    Penalty grows near joint limits.
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


def compute_position_cost(fk, joint_names, q_np, g_np, Qp, warn_prefix="fk_pos"):
    """
    Computes FK-based position running or terminal cost.
    """
    n = q_np.shape[0]
    cost_np = np.zeros((n,), dtype=np.float64)

    for i in range(n):
        try:
            p = fk.ee_position(joint_names, q_np[i])
            e = p - g_np[i]
            cost_np[i] = Qp * float(np.dot(e, e))
        except Exception as e:
            rospy.logwarn("%s: couldn't retrieve FK position: %s", warn_prefix, e)
            cost_np[i] = 1e3

    return cost_np


# ------------------------------------------------------------
# MoveIt state-validity helpers
# ------------------------------------------------------------

def is_state_valid(svc, active_joints, q_row, group_name):
    """
    Check one joint state using MoveIt's /check_state_validity.
    """
    req = GetStateValidityRequest()
    req.group_name = group_name
    req.robot_state = robot_state_from_q(active_joints, q_row)

    try:
        resp = svc(req)
        return bool(resp.valid)
    except Exception as e:
        rospy.logwarn("MoveIt state-validity service call failed: %s", e)
        return False


def filter_valid_qtg(
    svc,
    active_joints,
    group_name,
    q_np,
    t_np,
    g_np,
):
    """
    Keep only samples whose q state is valid in MoveIt.
    """
    q_keep, t_keep, g_keep = [], [], []

    for i in range(q_np.shape[0]):
        if is_state_valid(svc, active_joints, q_np[i], group_name):
            q_keep.append(q_np[i])
            t_keep.append(t_np[i])
            g_keep.append(g_np[i])

    if len(q_keep) == 0:
        return (
            np.empty((0, 7), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    return (
        np.asarray(q_keep, dtype=np.float64),
        np.asarray(t_keep, dtype=np.float64).reshape(-1, 1),
        np.asarray(g_keep, dtype=np.float64).reshape(-1, 3),
    )


# ------------------------------------------------------------
# GLF-style adaptive sampler
# ------------------------------------------------------------

class GLFSampler:
    """
    Global-Local Fusion style sampler.

    Global part:
        Uniformly samples q, t, g.

    Local part:
        Samples from a residual-prioritized buffer and perturbs selected points.

    Buffer entries are:
        {
            "q": np.ndarray shape (7,),
            "t": np.ndarray shape (1,),
            "g": np.ndarray shape (3,),
            "score": float
        }
    """

    def __init__(
        self,
        jmin,
        jmax,
        T,
        batch_size,
        global_frac=0.7,
        buffer_size=5000,
        sigma_q=0.08,
        sigma_t=0.10,
        sigma_g=0.03,
        local_score_sharpness=5.0,
    ):
        self.jmin = np.asarray(jmin, dtype=np.float64)
        self.jmax = np.asarray(jmax, dtype=np.float64)
        self.T = float(T)
        self.batch_size = int(batch_size)

        self.global_frac = float(global_frac)
        self.global_frac = float(np.clip(self.global_frac, 0.1, 1.0))

        self.buffer_size = int(buffer_size)
        self.sigma_q = float(sigma_q)
        self.sigma_t = float(sigma_t)
        self.sigma_g = float(sigma_g)
        self.local_score_sharpness = float(local_score_sharpness)

        self.buffer = []

    def sample_global(self, n):
        q = np.random.uniform(self.jmin, self.jmax, (n, 7)).astype(np.float64)
        t = np.random.uniform(0.0, self.T, (n, 1)).astype(np.float64)
        g = sample_goals(n)
        return q, t, g

    def sample_local(self, n):
        if n <= 0:
            return (
                np.empty((0, 7), dtype=np.float64),
                np.empty((0, 1), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64),
            )

        if len(self.buffer) == 0:
            return self.sample_global(n)

        scores = np.array([b["score"] for b in self.buffer], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=0.0)
        scores = np.maximum(scores, 0.0)

        if scores.sum() <= 0.0:
            probs = None
        else:
            probs = scores + 1e-8
            probs = probs / probs.sum()

        idx = np.random.choice(len(self.buffer), size=n, replace=True, p=probs)

        q_list, t_list, g_list = [], [], []

        score_max = float(scores.max() + 1e-8)

        for k in idx:
            item = self.buffer[k]

            score_norm = float(item["score"]) / score_max
            score_norm = np.clip(score_norm, 0.0, 1.0)

            # Higher score gets tighter local perturbation.
            scale = 1.0 / (1.0 + self.local_score_sharpness * score_norm)

            q_new = item["q"] + np.random.normal(
                loc=0.0,
                scale=self.sigma_q * scale,
                size=(7,),
            )

            t_new = item["t"] + np.random.normal(
                loc=0.0,
                scale=self.sigma_t * self.T * scale,
                size=(1,),
            )

            g_new = item["g"] + np.random.normal(
                loc=0.0,
                scale=self.sigma_g * scale,
                size=(3,),
            )

            q_new = np.clip(q_new, self.jmin, self.jmax)
            t_new = np.clip(t_new, 0.0, self.T)
            g_new = clip_goals(g_new)

            q_list.append(q_new)
            t_list.append(t_new)
            g_list.append(g_new)

        return (
            np.asarray(q_list, dtype=np.float64).reshape(n, 7),
            np.asarray(t_list, dtype=np.float64).reshape(n, 1),
            np.asarray(g_list, dtype=np.float64).reshape(n, 3),
        )

    def sample_raw_batch(self, batch_size=None):
        """
        Generate a mixed global/local batch before MoveIt filtering.
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = int(batch_size)

        n_global = int(round(batch_size * self.global_frac))
        n_global = np.clip(n_global, 1, batch_size)
        n_local = batch_size - n_global

        q_g, t_g, goal_g = self.sample_global(n_global)
        q_l, t_l, goal_l = self.sample_local(n_local)

        q = np.vstack([q_g, q_l])
        t = np.vstack([t_g, t_l])
        g = np.vstack([goal_g, goal_l])

        return q, t, g

    def sample_valid_batch(
        self,
        svc,
        active_joints,
        group_name,
        batch_size=None,
        max_attempt_factor=25,
        candidate_multiplier=2,
    ):
        """
        Generate a GLF mixed batch and reject invalid MoveIt states.
        """
        if batch_size is None:
            batch_size = self.batch_size

        batch_size = int(batch_size)
        q_all, t_all, g_all = [], [], []

        attempts = 0
        max_attempts = max_attempt_factor * batch_size

        while len(q_all) < batch_size and attempts < max_attempts:
            n_needed = batch_size - len(q_all)
            n_candidates = max(candidate_multiplier * n_needed, 8)

            q_c, t_c, g_c = self.sample_raw_batch(n_candidates)

            q_v, t_v, g_v = filter_valid_qtg(
                svc=svc,
                active_joints=active_joints,
                group_name=group_name,
                q_np=q_c,
                t_np=t_c,
                g_np=g_c,
            )

            attempts += q_c.shape[0]

            for i in range(q_v.shape[0]):
                q_all.append(q_v[i])
                t_all.append(t_v[i])
                g_all.append(g_v[i])

                if len(q_all) >= batch_size:
                    break

        if len(q_all) < batch_size:
            raise RuntimeError(
                f"Could only collect {len(q_all)}/{batch_size} valid samples "
                f"after {attempts} candidate checks."
            )

        return (
            np.asarray(q_all, dtype=np.float64).reshape(batch_size, 7),
            np.asarray(t_all, dtype=np.float64).reshape(batch_size, 1),
            np.asarray(g_all, dtype=np.float64).reshape(batch_size, 3),
        )

    def update_buffer(self, q_np, t_np, g_np, scores_np, top_k=64):
        """
        Add highest-residual samples to local adaptive buffer.
        """
        if q_np.shape[0] == 0:
            return

        scores_np = np.asarray(scores_np, dtype=np.float64).reshape(-1)
        scores_np = np.nan_to_num(scores_np, nan=0.0, posinf=1e6, neginf=0.0)
        scores_np = np.maximum(scores_np, 0.0)

        n = min(q_np.shape[0], scores_np.shape[0])
        top_k = min(int(top_k), n)

        if top_k <= 0:
            return

        idx = np.argsort(scores_np[:n])[-top_k:]

        for i in idx:
            self.buffer.append(
                {
                    "q": q_np[i].copy(),
                    "t": t_np[i].copy(),
                    "g": g_np[i].copy(),
                    "score": float(scores_np[i]),
                }
            )

        if len(self.buffer) > self.buffer_size:
            self.buffer = sorted(
                self.buffer,
                key=lambda x: x["score"],
                reverse=True,
            )[: self.buffer_size]


# Main training entry point

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

    Cj = float(rospy.get_param("~Cj", 0.01))
    Cv = float(rospy.get_param("~Cv", 0.001))
    Ctr = float(rospy.get_param("~Ctr", 100.0))

    use_velocity_loss = bool(rospy.get_param("~use_velocity_loss", False))

    R_diag = torch.tensor(
        rospy.get_param("~R_diag", [0.15] * 7),
        dtype=torch.float32,
        device=device,
    )
    R_inv_diag = 1.0 / R_diag

    print(f"R_diag {R_diag}")

    vel_limits_np = np.array(
        rospy.get_param("~vel_limits", [2.0] * 7),
        dtype=np.float64,
    )
    vel_limits = torch.tensor(vel_limits_np, dtype=torch.float32, device=device)

    out_path = rospy.get_param(
        "~out_path",
        "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pth",
    )

    train_perf_data_path = rospy.get_param(
        "~train_perf_path",
        "/root/catkin_ws/src/object_tracking/models/train_perf_data.csv"
    )

    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")

    group = MoveGroupCommander(group_name)
    joint_names = group.get_active_joints()

    if len(joint_names) != 7:
        raise RuntimeError(f"Expected 7 joints, got {len(joint_names)}: {joint_names}")

    fk = FKClient(service="/compute_fk", ee_link=ee_link, frame=world_frame)
    jmin, jmax = panda_joint_limits()

    state_validity_service = rospy.get_param(
        "~state_validity_service",
        "/check_state_validity",
    )

    rospy.loginfo("Waiting for MoveIt state-validity service: %s", state_validity_service)
    rospy.wait_for_service(state_validity_service)
    validity_svc = rospy.ServiceProxy(state_validity_service, GetStateValidity)

    # Model
    model_type = rospy.get_param("~model_type", "ValueNet")

    if model_type == "DGMValueNet":
        model = DGMValueNet(in_dim=11, hidden=hidden, depth=depth).to(device)
    elif model_type == "ValueNet_":
        model = ValueNet_(
            num_layers=8,
            input_dim=11,
            output_dim=1,
            hidden_size=192,
            expansion_factor=1,
        ).to(device)
    else:
        model = ValueNet(
            num_layers=8,
            input_dim=11,
            output_dim=1,
            hidden_size=192,
        ).to(device)

    opt = optim.Adam(model.parameters(), lr=lr)

    # GLF sampler params
    sampler = GLFSampler(
        jmin=jmin,
        jmax=jmax,
        T=T,
        batch_size=batch,
        global_frac=float(rospy.get_param("~global_frac", 0.7)),
        buffer_size=int(rospy.get_param("~glf_buffer_size", 5000)),
        sigma_q=float(rospy.get_param("~glf_sigma_q", 0.08)),
        sigma_t=float(rospy.get_param("~glf_sigma_t", 0.10)),
        sigma_g=float(rospy.get_param("~glf_sigma_g", 0.03)),
        local_score_sharpness=float(rospy.get_param("~glf_local_score_sharpness", 5.0)),
    )

    glf_warmup_iters = int(rospy.get_param("~glf_warmup_iters", 50))
    glf_top_k = int(rospy.get_param("~glf_top_k", min(64, batch // 3)))

    t0 = time.time()
    loss = torch.tensor(0.0, device=device)

    t_loss = 0.0
    t_loss_pde = 0.0
    t_loss_term = 0.0

    log_every = int(rospy.get_param("~log_every", 10))

    now = str(datetime.now()).replace("-", "").replace(" ", ":")

    results_file = now + ".csv"

    results_preamble = (
        str(model)
        + "\n"
        + f"lr:{lr}\n"
        + f"Cj:{Cj},Cv:{Cv},Ctr:{Ctr}\n"
        + f"global_frac:{sampler.global_frac},buffer_size:{sampler.buffer_size}"
    )

    print(f"results_file {results_file}\n\nresults_preamble {results_preamble}")
    data_header = f"time,t_loss_pde,t_loss_term,t_loss\n"

    os.makedirs(os.path.dirname(train_perf_data_path), exist_ok=True)

    with open(train_perf_data_path, "a") as f:
        f.write(results_preamble)

    for it in range(1, iters + 1):
        # During warmup, force mostly global sampling.
        original_global_frac = sampler.global_frac
        if it <= glf_warmup_iters:
            sampler.global_frac = 1.0

        q_np, t_np, g_np = sampler.sample_valid_batch(
            svc=validity_svc,
            active_joints=joint_names,
            group_name=group_name,
            batch_size=batch,
        )

        sampler.global_frac = original_global_frac

        # Optional audit using your rollout-style validity checker.
        ok, first_bad_idx, msg = validate_with_moveit_state_validity(
            svc=validity_svc,
            active_joints=joint_names,
            q_hist=q_np,
            group_name=group_name,
            stride=1,
        )

        if not ok:
            rospy.logwarn(
                "Unexpected invalid state after filtering at index %d: %s. Resampling.",
                first_bad_idx,
                msg,
            )
            q_np, t_np, g_np = sampler.sample_valid_batch(
                svc=validity_svc,
                active_joints=joint_names,
                group_name=group_name,
                batch_size=batch,
            )

        # Running cost
        pos_cost_np = compute_position_cost(
            fk=fk,
            joint_names=joint_names,
            q_np=q_np,
            g_np=g_np,
            Qp=Qp,
            warn_prefix="fk_pos l",
        )

        joint_limit_penalty_np = batch_joint_limit_penalty(q_np, jmin, jmax)

        # Velocity-limit penalty, if used, is added inside hjb_residual_loss_.
        l_np = pos_cost_np + Cj * joint_limit_penalty_np

        q = torch.tensor(q_np, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor(t_np / T, dtype=torch.float32, device=device, requires_grad=True)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)
        l = torch.tensor(l_np, dtype=torch.float32, device=device)

        x = build_input(q, t, g)

        # Your ValueNet currently appears to expect two identical inputs.
        # DGMValueNet may expect only one input depending on your implementation.
        if model_type == "DGMValueNet":
            V = model(x)
        else:
            V = model(x, x)

        if use_velocity_loss:
            loss_pde, residual_vec = hjb_residual_(
                V=V,
                q=q,
                t_norm=t,
                running_cost=l,
                R_inv_diag=R_inv_diag,
                vel_limits=vel_limits,
                Cv=Cv,
                reduction="mean",
                return_residual=True,
            )
        else:
            loss_pde, residual_vec = hjb_residual(
                V=V,
                q=q,
                t_norm=t,
                running_cost=l,
                R_inv_diag=R_inv_diag,
                reduction="mean",
                return_residual=True,
            )

        residual_scores_np = residual_vec.detach().abs().cpu().numpy()

        # Terminal batch
        bt = max(64, batch // 3)

        qT_np, tT_phys_np, gT_np = sampler.sample_valid_batch(
            svc=validity_svc,
            active_joints=joint_names,
            group_name=group_name,
            batch_size=bt,
        )

        # Terminal time is fixed to normalized 1.0.
        phi_np = compute_position_cost(
            fk=fk,
            joint_names=joint_names,
            q_np=qT_np,
            g_np=gT_np,
            Qp=QpT,
            warn_prefix="fk_pos phi",
        )

        qT = torch.tensor(qT_np, dtype=torch.float32, device=device)
        tT = torch.ones((bt, 1), dtype=torch.float32, device=device, requires_grad=True)
        gT = torch.tensor(gT_np, dtype=torch.float32, device=device)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=device)

        xT = build_input(qT, tT, gT)

        if model_type == "DGMValueNet":
            VT = model(xT)
        else:
            VT = model(xT, xT)

        loss_term = terminal_loss(VT, phi)

        loss = loss_pde + Ctr * loss_term

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        # Update GLF buffer after the optimizer step.
        sampler.update_buffer(
            q_np=q_np,
            t_np=t_np,
            g_np=g_np,
            scores_np=residual_scores_np,
            top_k=glf_top_k,
        )

        t_loss_pde += float(loss_pde.item())
        t_loss_term += float(loss_term.item())
        t_loss += float(loss.item())

        if it % log_every == 0:
            rospy.loginfo(
                (
                    "iter=%d pde=%.6f term=%.6f loss=%.6f "
                    "buffer=%d global_frac=%.2f elapsed=%.1fs"
                ),
                it,
                t_loss_pde / log_every,
                t_loss_term / log_every,
                t_loss / log_every,
                len(sampler.buffer),
                sampler.global_frac,
                time.time() - t0,
            )

            data_line = (f"{it}"
                         f",{float(t_loss_pde) / (log_every * (batch + bt))}"
                         f",{float(t_loss_term) / (log_every * (batch + bt))}"
                         f",{float(t_loss) / (log_every * (batch + bt))}"
                         )

            with open(train_perf_data_path, "a") as f:
                f.write(data_line + "\n")

            t_loss = 0.0
            t_loss_pde = 0.0
            t_loss_term = 0.0

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    checkpoint = {
        "epoch": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "loss": float(loss.item()),
        "model_type": model_type,
        "T": T,
        "R_diag": R_diag.detach().cpu().numpy(),
        "joint_names": joint_names,
        "jmin": jmin,
        "jmax": jmax,
        "sampler": {
            "global_frac": sampler.global_frac,
            "buffer_size": sampler.buffer_size,
            "sigma_q": sampler.sigma_q,
            "sigma_t": sampler.sigma_t,
            "sigma_g": sampler.sigma_g,
        },
    }

    torch.save(checkpoint, out_path)
    f.close()

    rospy.loginfo("Saved epoch checkpoint: %s", out_path)
    rospy.loginfo("DONE. Saved final: %s", out_path)


if __name__ == "__main__":
    main_()