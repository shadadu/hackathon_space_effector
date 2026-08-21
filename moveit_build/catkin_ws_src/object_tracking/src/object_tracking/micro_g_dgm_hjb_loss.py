from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp

from .dgm_jax import apply_value


Params = List[Dict[str, jnp.ndarray]]
AdamState = Dict[str, Any]


def build_input(q, b, r, v_o, tau):
    return jnp.concatenate([q, b, r, v_o, tau], axis=-1)


def value_scalar(params: Params, q, b, r, v_o, tau):
    q = jnp.asarray(q, dtype=jnp.float32).reshape(1, 7)
    b = jnp.asarray(b, dtype=jnp.float32).reshape(1, 3)
    r = jnp.asarray(r, dtype=jnp.float32).reshape(1, 3)
    v_o = jnp.asarray(v_o, dtype=jnp.float32).reshape(1, 3)
    tau = jnp.asarray(tau, dtype=jnp.float32).reshape(1, 1)
    return apply_value(params, build_input(q, b, r, v_o, tau))[0]


def value_and_grads(params: Params, q, b, r, v_o, tau):
    grad_fn = jax.value_and_grad(value_scalar, argnums=(1, 2, 3, 5))
    return grad_fn(params, q, b, r, v_o, tau)


value_and_grads_jit = jax.jit(value_and_grads)


def entry_gain(r, goal_pos_tol, entry_guard_width, entry_velocity_weight):
    distance = jnp.linalg.norm(r, axis=1)
    blend = jnp.clip(
        (goal_pos_tol + entry_guard_width - distance) / entry_guard_width,
        0.0,
        1.0,
    )
    return entry_velocity_weight * blend


def policy_terms(
        dv_dq,
        dv_db,
        dv_dr,
        jac_ee,
        r_q_diag,
        r_b_diag,
        r,
        v_o,
        goal_pos_tol,
        entry_guard_width,
        entry_velocity_weight,
):
    """Minimize the quadratic Hamiltonian including near-goal velocity matching."""
    a_q = dv_dq + jnp.einsum("bij,bj->bi", jnp.swapaxes(jac_ee, 1, 2), dv_dr)
    a_b = dv_db + dv_dr
    linear_term = jnp.concatenate([a_q, a_b], axis=1)

    batch_size = jac_ee.shape[0]
    eye_3 = jnp.broadcast_to(jnp.eye(3, dtype=jac_ee.dtype), (batch_size, 3, 3))
    velocity_map = jnp.concatenate([jac_ee, eye_3], axis=2)
    gain = entry_gain(
        r, goal_pos_tol, entry_guard_width, entry_velocity_weight
    )

    control_diag = jnp.concatenate([r_q_diag.reshape(7), r_b_diag.reshape(3)])
    control_metric = jnp.broadcast_to(jnp.diag(control_diag), (batch_size, 10, 10))
    velocity_map_t = jnp.swapaxes(velocity_map, 1, 2)
    lhs = control_metric + gain[:, None, None] * jnp.einsum(
        "bij,bjk->bik", velocity_map_t, velocity_map
    )
    rhs = -linear_term + gain[:, None] * jnp.einsum(
        "bij,bj->bi", velocity_map_t, v_o
    )
    controls = jnp.linalg.solve(lhs, rhs[..., None])[..., 0]
    return controls[:, :7], controls[:, 7:], gain


def value_grads_batch(params: Params, batch: Dict[str, jnp.ndarray], suffix: str = ""):
    def v_for_state(q_i, b_i, r_i, v_i, tau_i):
        return value_scalar(params, q_i, b_i, r_i, v_i, tau_i)

    args = (
        batch["q" + suffix],
        batch["b" + suffix],
        batch["r" + suffix],
        batch["v_o" + suffix],
        batch["tau" + suffix],
    )
    dv_dq = jax.vmap(jax.grad(v_for_state, argnums=0))(*args)
    dv_db = jax.vmap(jax.grad(v_for_state, argnums=1))(*args)
    dv_dr = jax.vmap(jax.grad(v_for_state, argnums=2))(*args)
    dv_dtau = jax.vmap(jax.grad(v_for_state, argnums=4))(*args).reshape(-1)
    return dv_dq, dv_db, dv_dr, dv_dtau


def timeout_loss(params: Params, batch: Dict[str, jnp.ndarray]):
    v_t = apply_value(
        params,
        build_input(
            batch["q_t"],
            batch["b_t"],
            batch["r_t"],
            batch["v_o_t"],
            batch["tau_t"],
        ),
    )
    return jnp.mean((v_t.reshape(-1) - batch["phi_t"].reshape(-1)) ** 2)


def goal_loss(params: Params, batch: Dict[str, jnp.ndarray]):
    """Apply V=0 only where the derived action also meets the velocity guard."""
    v_g = apply_value(
        params,
        build_input(
            batch["q_g"],
            batch["b_g"],
            batch["r_g"],
            batch["v_o_g"],
            batch["tau_g"],
        ),
    )
    dv_dq, dv_db, dv_dr, _ = value_grads_batch(params, batch, "_g")
    u_q, u_b, _ = policy_terms(
        dv_dq,
        dv_db,
        dv_dr,
        batch["jac_ee_g"],
        batch["r_q_diag"],
        batch["r_b_diag"],
        batch["r_g"],
        batch["v_o_g"],
        batch["goal_pos_tol"],
        batch["entry_guard_width"],
        batch["entry_velocity_weight"],
    )
    v_rel = jnp.einsum("bij,bj->bi", batch["jac_ee_g"], u_q) + u_b - batch["v_o_g"]
    ready = (
        jnp.linalg.norm(v_rel, axis=1) <= batch["goal_vel_tol"]
    ).astype(v_g.dtype)
    value_error = (v_g.reshape(-1) - batch["phi_g"].reshape(-1)) ** 2
    restricted_value_loss = jnp.sum(ready * value_error) / jnp.maximum(jnp.sum(ready), 1.0)
    entry_violation = jnp.maximum(
        0.0, jnp.linalg.norm(v_rel, axis=1) - batch["goal_vel_tol"]
    )
    return restricted_value_loss + batch["entry_velocity_weight"] * jnp.mean(entry_violation ** 2)


def hjb_residual_loss(params: Params, batch: Dict[str, jnp.ndarray]):
    dv_dq, dv_db, dv_dr, dv_dtau = value_grads_batch(params, batch)

    u_q, u_b, gain = policy_terms(
        dv_dq,
        dv_db,
        dv_dr,
        batch["jac_ee"],
        batch["r_q_diag"],
        batch["r_b_diag"],
        batch["r"],
        batch["v_o"],
        batch["goal_pos_tol"],
        batch["entry_guard_width"],
        batch["entry_velocity_weight"],
    )

    r_dot = jnp.einsum("bij,bj->bi", batch["jac_ee"], u_q) + u_b - batch["v_o"]
    control_cost = 0.5 * jnp.sum((u_q ** 2) * batch["r_q_diag"].reshape(1, 7), axis=1)
    control_cost += 0.5 * jnp.sum((u_b ** 2) * batch["r_b_diag"].reshape(1, 3), axis=1)
    control_cost += 0.5 * gain * jnp.sum(r_dot ** 2, axis=1)

    drift_term = jnp.sum(dv_dq * u_q, axis=1)
    drift_term += jnp.sum(dv_db * u_b, axis=1)
    drift_term += jnp.sum(dv_dr * r_dot, axis=1)

    residual = -dv_dtau + batch["running_cost"].reshape(-1) + control_cost + drift_term
    position_ready = jnp.linalg.norm(batch["r"], axis=1) <= batch["goal_pos_tol"]
    velocity_ready = jnp.linalg.norm(r_dot, axis=1) <= batch["goal_vel_tol"]
    continuation = jnp.logical_not(jnp.logical_and(position_ready, velocity_ready)).astype(residual.dtype)
    return jnp.sum(continuation * residual ** 2) / jnp.maximum(jnp.sum(continuation), 1.0)


def total_loss(params: Params, batch: Dict[str, jnp.ndarray], cpd: float, ctr: float, cgoal: float):
    loss_pde = hjb_residual_loss(params, batch)
    loss_timeout = timeout_loss(params, batch)
    loss_goal = goal_loss(params, batch)
    loss = cpd * loss_pde + ctr * loss_timeout + cgoal * loss_goal
    return loss, (loss_pde, loss_timeout, loss_goal)


def adam_init(params: Params) -> AdamState:
    return {
        "step": jnp.array(0, dtype=jnp.int32),
        "m": jax.tree_util.tree_map(jnp.zeros_like, params),
        "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    }


def adam_update(params: Params, grads: Params, state: AdamState, lr: float, clip_norm: float = 5.0):
    step = state["step"] + 1
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)))
    scale = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-6))
    grads = jax.tree_util.tree_map(lambda g: g * scale, grads)

    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    m = jax.tree_util.tree_map(lambda m_i, g: beta1 * m_i + (1.0 - beta1) * g, state["m"], grads)
    v = jax.tree_util.tree_map(lambda v_i, g: beta2 * v_i + (1.0 - beta2) * (g ** 2), state["v"], grads)
    beta1_power = jnp.power(beta1, step)
    beta2_power = jnp.power(beta2, step)
    m_hat = jax.tree_util.tree_map(lambda m_i: m_i / (1.0 - beta1_power), m)
    v_hat = jax.tree_util.tree_map(lambda v_i: v_i / (1.0 - beta2_power), v)
    params = jax.tree_util.tree_map(
        lambda p, m_i, v_i: p - lr * m_i / (jnp.sqrt(v_i) + eps),
        params,
        m_hat,
        v_hat,
    )
    return params, {"step": step, "m": m, "v": v}


@jax.jit
def train_step(
        params: Params,
        opt_state: AdamState,
        batch: Dict[str, jnp.ndarray],
        lr: float,
        cpd: float,
        ctr: float,
        cgoal: float,
        clip_norm: float,
):
    (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
        params, batch, cpd, ctr, cgoal
    )
    params, opt_state = adam_update(params, grads, opt_state, lr, clip_norm)
    return params, opt_state, loss, aux[0], aux[1], aux[2]


def make_batch(
        q_np,
        b_np,
        r_np,
        v_o_np,
        tau_np,
        jac_ee_np,
        running_cost_np,
        q_t_np,
        b_t_np,
        r_t_np,
        v_o_t_np,
        tau_t_np,
        phi_t_np,
        q_g_np,
        b_g_np,
        r_g_np,
        v_o_g_np,
        tau_g_np,
        phi_g_np,
        jac_ee_g_np,
        goal_pos_tol,
        goal_vel_tol,
        entry_guard_width,
        entry_velocity_weight,
        r_q_diag_np,
        r_b_diag_np,
):
    r_q_diag = jnp.asarray(r_q_diag_np, dtype=jnp.float32)
    r_b_diag = jnp.asarray(r_b_diag_np, dtype=jnp.float32)
    return {
        "q": jnp.asarray(q_np, dtype=jnp.float32),
        "b": jnp.asarray(b_np, dtype=jnp.float32),
        "r": jnp.asarray(r_np, dtype=jnp.float32),
        "v_o": jnp.asarray(v_o_np, dtype=jnp.float32),
        "tau": jnp.asarray(tau_np, dtype=jnp.float32),
        "jac_ee": jnp.asarray(jac_ee_np, dtype=jnp.float32),
        "running_cost": jnp.asarray(running_cost_np, dtype=jnp.float32),
        "q_t": jnp.asarray(q_t_np, dtype=jnp.float32),
        "b_t": jnp.asarray(b_t_np, dtype=jnp.float32),
        "r_t": jnp.asarray(r_t_np, dtype=jnp.float32),
        "v_o_t": jnp.asarray(v_o_t_np, dtype=jnp.float32),
        "tau_t": jnp.asarray(tau_t_np, dtype=jnp.float32),
        "phi_t": jnp.asarray(phi_t_np, dtype=jnp.float32),
        "q_g": jnp.asarray(q_g_np, dtype=jnp.float32),
        "b_g": jnp.asarray(b_g_np, dtype=jnp.float32),
        "r_g": jnp.asarray(r_g_np, dtype=jnp.float32),
        "v_o_g": jnp.asarray(v_o_g_np, dtype=jnp.float32),
        "tau_g": jnp.asarray(tau_g_np, dtype=jnp.float32),
        "phi_g": jnp.asarray(phi_g_np, dtype=jnp.float32),
        "jac_ee_g": jnp.asarray(jac_ee_g_np, dtype=jnp.float32),
        "goal_pos_tol": jnp.asarray(goal_pos_tol, dtype=jnp.float32),
        "goal_vel_tol": jnp.asarray(goal_vel_tol, dtype=jnp.float32),
        "entry_guard_width": jnp.asarray(entry_guard_width, dtype=jnp.float32),
        "entry_velocity_weight": jnp.asarray(entry_velocity_weight, dtype=jnp.float32),
        "r_q_diag": r_q_diag,
        "r_b_diag": r_b_diag,
        "r_q_inv_diag": 1.0 / jnp.maximum(r_q_diag, 1e-9),
        "r_b_inv_diag": 1.0 / jnp.maximum(r_b_diag, 1e-9),
    }


def policy_np(
        params: Params,
        q,
        b,
        r,
        v_o,
        tau,
        jac_ee,
        r_q_diag,
        r_b_diag,
        goal_pos_tol,
        entry_guard_width,
        entry_velocity_weight,
):
    (_, grads) = value_and_grads_jit(params, q, b, r, v_o, tau)
    dv_dq, dv_db, dv_dr, _ = grads
    u_q, u_b, _ = policy_terms(
        dv_dq.reshape(1, 7),
        dv_db.reshape(1, 3),
        dv_dr.reshape(1, 3),
        jnp.asarray(jac_ee, dtype=jnp.float32).reshape(1, 3, 7),
        jnp.asarray(r_q_diag, dtype=jnp.float32),
        jnp.asarray(r_b_diag, dtype=jnp.float32),
        jnp.asarray(r, dtype=jnp.float32).reshape(1, 3),
        jnp.asarray(v_o, dtype=jnp.float32).reshape(1, 3),
        jnp.asarray(goal_pos_tol, dtype=jnp.float32),
        jnp.asarray(entry_guard_width, dtype=jnp.float32),
        jnp.asarray(entry_velocity_weight, dtype=jnp.float32),
    )
    return u_q[0], u_b[0]
