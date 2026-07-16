#!/usr/bin/env python3
import os
import pickle
import time
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import rospy
from moveit_commander import MoveGroupCommander
from moveit_msgs.srv import GetStateValidity

from object_tracking.dgm_jax import init_mlp_params, load_checkpoint, save_checkpoint
from object_tracking.fk_client import FKClient
from object_tracking.micro_g_dgm_hjb_loss import adam_init, make_batch, train_step


from moveit_msgs.msg import RobotState
from sensor_msgs.msg import JointState

def robot_state_from_q(active_joints, q):
    state = RobotState()
    state.joint_state = JointState(
        name=list(active_joints),
        position=[float(x) for x in q],
    )
    return state

def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def sample_box(n, lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.random.uniform(lo, hi, (n, lo.shape[0])).astype(np.float64)

def sample_const_velocities(n, lo, hi):
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    # rospy.loginfo("Sampling constant velocities: lo=%s, hi=%s, n=%s, lo.shape=%s", lo, hi, n, lo.shape[0])
    return np.tile(np.random.uniform(lo, hi, lo.shape[0]), (n, 1)).astype(np.float64)


def is_state_valid(svc, active_joints, q_row, group_name):
    req = type("Req", (), {})()
    try:
        from moveit_msgs.srv import GetStateValidityRequest
        req = GetStateValidityRequest()
        req.group_name = group_name
        req.robot_state = robot_state_from_q(active_joints, q_row)
        return bool(svc(req).valid)
    except Exception as exc:
        rospy.logwarn_throttle(2.0, "State validity check failed; accepting sampled state for now: %s", exc)
        return True


def sample_q_batch(validity_svc, active_joints, group_name, jmin, jmax, batch_size, use_validity):
    if not use_validity:
        return np.random.uniform(jmin, jmax, (batch_size, 7)).astype(np.float64)

    rows = []
    attempts = 0
    max_attempts = max(100, batch_size * 30)
    while len(rows) < batch_size and attempts < max_attempts:
        q = np.random.uniform(jmin, jmax, (7,)).astype(np.float64)
        attempts += 1
        if is_state_valid(validity_svc, active_joints, q, group_name):
            rows.append(q)
    if len(rows) < batch_size:
        raise RuntimeError(f"Only sampled {len(rows)}/{batch_size} valid q states")
    return np.asarray(rows, dtype=np.float64)


def jacobian_batch(group, q_np):
    out = np.zeros((q_np.shape[0], 3, 7), dtype=np.float64)
    for i in range(q_np.shape[0]):
        jac = np.asarray(group.get_jacobian_matrix([float(x) for x in q_np[i].tolist()]), dtype=np.float64)
        if jac.shape[0] < 3:
            raise RuntimeError(f"Jacobian shape {jac.shape} does not include translational rows")
        out[i, :, :] = jac[:3, :7]
    return out


def fk_position_batch(fk_client, active_joints, q_np):
    out = np.zeros((q_np.shape[0], 3), dtype=np.float64)
    for i in range(q_np.shape[0]):
        out[i, :] = fk_client.ee_position(active_joints, q_np[i])
    return out


def reachability_shell_cost(p_ee_local_np, r_np, reach_min, reach_max, weight):
    if weight <= 0.0:
        return np.zeros((r_np.shape[0],), dtype=np.float64)
    object_from_base = p_ee_local_np - r_np
    dist = np.linalg.norm(object_from_base, axis=1)
    near = np.maximum(0.0, reach_min - dist)
    far = np.maximum(0.0, dist - reach_max)
    return 0.5 * weight * (near * near + far * far)


def load_pretrain(path):
    model, meta = load_checkpoint(path)
    opt_state = adam_init(model)
    try:
        with Path(path).open("rb") as f:
            ckpt = pickle.load(f)
        if "opt_state" in ckpt:
            opt_state = jax.tree_util.tree_map(jnp.asarray, ckpt["opt_state"])
    except Exception as exc:
        rospy.logwarn("Could not restore optimizer state from %s: %s", path, exc)
    return model, opt_state, meta


def make_training_batch(
        group,
        validity_svc,
        active_joints,
        group_name,
        batch,
        bt,
        T,
        jmin,
        jmax,
        base_min,
        base_max,
        rel_min,
        rel_max,
        vo_min,
        vo_max,
        R_q_diag,
        R_b_diag,
        Qr,
        QrT,
        Qv,
        Qreach,
        QreachT,
        reach_min,
        reach_max,
        fk_client,
        use_validity,
):
    q_np = sample_q_batch(validity_svc, active_joints, group_name, jmin, jmax, batch, use_validity)
    b_np = sample_box(batch, base_min, base_max)
    r_np = sample_box(batch, rel_min, rel_max)
    v_o_np = sample_const_velocities(batch, vo_min, vo_max) # sample constant object velocities, consistent with the t/tau progression in the rollout
    tau_np = np.random.uniform(0.0, T, (batch, 1)).astype(np.float64)
    tau_np = np.sort(tau_np.flatten().reshape((batch, 1)))[::-1]
    jac_np = jacobian_batch(group, q_np)
    p_ee_local_np = fk_position_batch(fk_client, active_joints, q_np)

    running_cost_np = 0.5 * Qr * np.sum(r_np * r_np, axis=1)
    running_cost_np += reachability_shell_cost(p_ee_local_np, r_np, reach_min, reach_max, Qreach)

    q_t_np = sample_q_batch(validity_svc, active_joints, group_name, jmin, jmax, bt, use_validity)
    b_t_np = sample_box(bt, base_min, base_max)
    r_t_np = sample_box(bt, rel_min, rel_max)
    # v_o_t_np = sample_box(bt, vo_min, vo_max)
    v_o_t_np = v_o_np[np.random.choice(v_o_np.shape[0], size=bt, replace=True), :] # sample from the training batch to ensure consistency with the sampled object velocities
    tau_t_np = np.random.uniform(0.0, T, (bt, 1)).astype(np.float64)
    tau_t_np = np.sort(tau_t_np.flatten().reshape((bt, 1)))[::-1]
    p_ee_local_t_np = fk_position_batch(fk_client, active_joints, q_t_np)

    # Sample a few rows from the TC to add to the PDE batch, to help with training stability.
    # This is a bit hacky but seems to help.
    n_samples = 12
    indices = np.random.permutation(bt)[:n_samples]
    q_np[-n_samples:] = q_t_np[indices]
    b_t_np[-n_samples:] = b_t_np[indices]
    r_t_np[-n_samples:] = r_t_np[indices]
    v_o_t_np[-n_samples:] = v_o_t_np[indices]

    phi_t_np = 0.5 * QrT * np.sum(r_t_np * r_t_np, axis=1)
    phi_t_np += 0.5 * Qv * np.sum(v_o_t_np * v_o_t_np, axis=1)
    phi_t_np += reachability_shell_cost(p_ee_local_t_np, r_t_np, reach_min, reach_max, QreachT)

    return make_batch(
        q_np=q_np,
        b_np=b_np,
        r_np=r_np,
        v_o_np=v_o_np,
        tau_np=tau_np,
        jac_ee_np=jac_np,
        running_cost_np=running_cost_np,
        q_t_np=q_t_np,
        b_t_np=b_t_np,
        r_t_np=r_t_np,
        v_o_t_np=v_o_t_np,
        tau_t_np=tau_t_np,
        phi_t_np=phi_t_np,
        r_q_diag_np=R_q_diag,
        r_b_diag_np=R_b_diag,
    )


def main():
    rospy.init_node("micro_g_dgm_training")
    rospy.loginfo("JAX devices: %s", jax.devices())

    group_name = rospy.get_param("~group_name", "panda_arm")
    group = MoveGroupCommander(group_name)
    active_joints = group.get_active_joints()
    if len(active_joints) != 7:
        raise RuntimeError(f"Expected 7 active joints, got {len(active_joints)}: {active_joints}")

    T = float(rospy.get_param("~T", 2.0))
    iters = int(rospy.get_param("~iters", 5000))
    batch = int(rospy.get_param("~batch", 192))
    bt = int(rospy.get_param("~bt", max(64, batch // 3)))
    lr = float(rospy.get_param("~lr", 3e-4))
    hidden = int(rospy.get_param("~hidden", 192))
    depth = int(rospy.get_param("~depth", 24))
    seed = int(rospy.get_param("~seed", 0))
    load_model = as_bool(rospy.get_param("~load_model", False))
    out_path = rospy.get_param(
        "~out_path",
        "/root/catkin_ws/src/object_tracking/models/micro_g_dgm_v1.pkl",
    )
    pretrain_path = rospy.get_param("~pretrain_path", out_path)

    Qr = float(rospy.get_param("~Qr", 10.0))
    QrT = float(rospy.get_param("~Qr_terminal", 100.0))
    Qv = float(rospy.get_param("~Qv_terminal", 0.0))
    Qreach = float(rospy.get_param("~Qreach", 5.0))
    QreachT = float(rospy.get_param("~Qreach_terminal", 50.0))
    Cpd = float(rospy.get_param("~Cpd", 1e3))
    Ctr = float(rospy.get_param("~Ctr", 1e-2))

    R_q_diag = np.array(rospy.get_param("~R_q_diag", [0.15] * 7), dtype=np.float64)
    R_b_diag = np.array(rospy.get_param("~R_b_diag", [0.50] * 3), dtype=np.float64)

    jmin, jmax = panda_joint_limits()
    base_min = np.array(rospy.get_param("~base_min", [-0.50, -0.50, -0.20]), dtype=np.float64)
    base_max = np.array(rospy.get_param("~base_max", [0.50, 0.50, 0.50]), dtype=np.float64)
    rel_min = np.array(rospy.get_param("~rel_min", [-0.60, -0.60, -0.60]), dtype=np.float64)
    rel_max = np.array(rospy.get_param("~rel_max", [0.60, 0.60, 0.60]), dtype=np.float64)
    reach_min = float(rospy.get_param("~reach_min", 0.20))
    reach_max = float(rospy.get_param("~reach_max", 0.75))
    vo_min = np.array(rospy.get_param("~vo_min", [-0.05, -0.05, -0.02]), dtype=np.float64)
    vo_max = np.array(rospy.get_param("~vo_max", [0.05, 0.05, 0.02]), dtype=np.float64)

    fk_service = rospy.get_param("~fk_service", "/compute_fk")
    service_wait_timeout = float(rospy.get_param("~service_wait_timeout", 30.0))
    fk_client = FKClient(
        service=fk_service,
        ee_link=rospy.get_param("~ee_link", "panda_hand"),
        frame=rospy.get_param("~world_frame", "world"),
        timeout=service_wait_timeout,
    )

    use_validity = as_bool(rospy.get_param("~use_state_validity", True))
    validity_svc = None
    if use_validity:
        service = rospy.get_param("~state_validity_service", "/check_state_validity")
        try:
            rospy.wait_for_service(service, timeout=10.0)
            validity_svc = rospy.ServiceProxy(service, GetStateValidity)
        except Exception as exc:
            rospy.logwarn("State validity unavailable; sampling q without collision filtering: %s", exc)
            use_validity = False

    if load_model:
        model, opt_state, _ = load_pretrain(pretrain_path)
    else:
        model = init_mlp_params(jax.random.PRNGKey(seed), in_dim=17, hidden=hidden, depth=depth)
        opt_state = adam_init(model)

    train_perf_path = rospy.get_param(
        "~train_perf_path",
        "/root/catkin_ws/src/object_tracking/models/micro_g_train_perf_data.csv",
    )
    os.makedirs(os.path.dirname(train_perf_path), exist_ok=True)
    with open(train_perf_path, "a") as f:
        f.write("iter,loss_pde,loss_term,loss,elapsed_s\n")

    t0 = time.time()
    last_loss = 0.0
    log_every = int(rospy.get_param("~log_every", 50))
    iter_batch_size = (batch + bt)
    for it in range(1, iters + 1):
        train_batch = make_training_batch(
            group=group,
            validity_svc=validity_svc,
            active_joints=active_joints,
            group_name=group_name,
            batch=batch,
            bt=bt,
            T=T,
            jmin=jmin,
            jmax=jmax,
            base_min=base_min,
            base_max=base_max,
            rel_min=rel_min,
            rel_max=rel_max,
            vo_min=vo_min,
            vo_max=vo_max,
            R_q_diag=R_q_diag,
            R_b_diag=R_b_diag,
            Qr=Qr,
            QrT=QrT,
            Qv=Qv,
            Qreach=Qreach,
            QreachT=QreachT,
            reach_min=reach_min,
            reach_max=reach_max,
            fk_client=fk_client,
            use_validity=use_validity,
        )
        model, opt_state, loss, loss_pde, loss_term = train_step(model, opt_state, train_batch, lr, Cpd, Ctr)
        last_loss = float(loss)

        if it % log_every == 0:
            elapsed = time.time() - t0
            rospy.loginfo(
                "micro_g iter=%d pde=%.6f term=%.6f loss=%.6f elapsed=%.1fs",
                it, float(loss_pde)/iter_batch_size, float(loss_term)/iter_batch_size, last_loss/iter_batch_size, elapsed,
            )
            with open(train_perf_path, "a") as f:
                f.write(f"{it},{float(loss_pde)},{float(loss_term)},{last_loss},{elapsed}\n")

    meta = {
        "format": "micro_g_dgm_v1",
        "in_dim": 17,
        "hidden": hidden,
        "depth": depth,
        "T": T,
        "state": "q,b,r,v_o,tau",
        "controls": "joint_velocity,base_velocity",
        "reach_min": reach_min,
        "reach_max": reach_max,
        "Qreach": Qreach,
        "Qreach_terminal": QreachT,
        "framework": "jax",
    }
    save_checkpoint(out_path, model, opt_state, meta=meta, loss=last_loss)
    rospy.loginfo("Saved micro-g DGM checkpoint: %s", out_path)


if __name__ == "__main__":
    main()
