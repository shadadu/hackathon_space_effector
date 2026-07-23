from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp


Params = List[Dict[str, jnp.ndarray]]
AdamState = Dict[str, Any]


def build_input(q, b, r, v_o, tau):
    return jnp.concatenate([q, b, r, v_o, tau], axis=-1)


def apply_mlp(params: Params, x):
    y = x
    for layer in params[:-1]:
        y = jnp.tanh(jnp.matmul(y, layer["w"]) + layer["b"])
    y = jnp.matmul(y, params[-1]["w"]) + params[-1]["b"]
    return jnp.squeeze(y, axis=-1)


def value_scalar(params: Params, q, b, r, v_o, tau):
    q = jnp.asarray(q, dtype=jnp.float32).reshape(1, 7)
    b = jnp.asarray(b, dtype=jnp.float32).reshape(1, 3)
    r = jnp.asarray(r, dtype=jnp.float32).reshape(1, 3)
    v_o = jnp.asarray(v_o, dtype=jnp.float32).reshape(1, 3)
    tau = jnp.asarray(tau, dtype=jnp.float32).reshape(1, 1)
    return apply_mlp(params, build_input(q, b, r, v_o, tau))[0]


def value_and_grads(params: Params, q, b, r, v_o, tau):
    grad_fn = jax.value_and_grad(value_scalar, argnums=(1, 2, 3, 5))
    return grad_fn(params, q, b, r, v_o, tau)


value_and_grads_jit = jax.jit(value_and_grads)


def policy_terms(dv_dq, dv_db, dv_dr, jac_ee, r_q_inv_diag, r_b_inv_diag):
    a_q = dv_dq + jnp.einsum("bij,bj->bi", jnp.swapaxes(jac_ee, 1, 2), dv_dr)
    a_b = dv_db + dv_dr
    u_q = -r_q_inv_diag.reshape(1, 7) * a_q
    u_b = -r_b_inv_diag.reshape(1, 3) * a_b
    return u_q, u_b


def timeout_loss(params: Params, batch: Dict[str, jnp.ndarray]):
    v_t = apply_mlp(
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
    """Dirichlet loss on the absorbing first-exit goal set."""
    v_g = apply_mlp(
        params,
        build_input(
            batch["q_g"],
            batch["b_g"],
            batch["r_g"],
            batch["v_o_g"],
            batch["tau_g"],
        ),
    )
    return jnp.mean((v_g.reshape(-1) - batch["phi_g"].reshape(-1)) ** 2)


def hjb_residual_loss(params: Params, batch: Dict[str, jnp.ndarray]):
    def v_for_state(q_i, b_i, r_i, v_i, tau_i):
        return value_scalar(params, q_i, b_i, r_i, v_i, tau_i)

    dv_dq = jax.vmap(jax.grad(v_for_state, argnums=0))(
        batch["q"], batch["b"], batch["r"], batch["v_o"], batch["tau"]
    )
    dv_db = jax.vmap(jax.grad(v_for_state, argnums=1))(
        batch["q"], batch["b"], batch["r"], batch["v_o"], batch["tau"]
    )
    dv_dr = jax.vmap(jax.grad(v_for_state, argnums=2))(
        batch["q"], batch["b"], batch["r"], batch["v_o"], batch["tau"]
    )
    dv_dtau = jax.vmap(jax.grad(v_for_state, argnums=4))(
        batch["q"], batch["b"], batch["r"], batch["v_o"], batch["tau"]
    ).reshape(-1)

    u_q, u_b = policy_terms(
        dv_dq,
        dv_db,
        dv_dr,
        batch["jac_ee"],
        batch["r_q_inv_diag"],
        batch["r_b_inv_diag"],
    )

    r_dot = jnp.einsum("bij,bj->bi", batch["jac_ee"], u_q) + u_b - batch["v_o"]
    control_cost = 0.5 * jnp.sum((u_q ** 2) * batch["r_q_diag"].reshape(1, 7), axis=1)
    control_cost += 0.5 * jnp.sum((u_b ** 2) * batch["r_b_diag"].reshape(1, 3), axis=1)

    drift_term = jnp.sum(dv_dq * u_q, axis=1)
    drift_term += jnp.sum(dv_db * u_b, axis=1)
    drift_term += jnp.sum(dv_dr * r_dot, axis=1)

    residual = -dv_dtau + batch["running_cost"].reshape(-1) + control_cost + drift_term
    return jnp.mean(residual ** 2)


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
        "r_q_diag": r_q_diag,
        "r_b_diag": r_b_diag,
        "r_q_inv_diag": 1.0 / jnp.maximum(r_q_diag, 1e-9),
        "r_b_inv_diag": 1.0 / jnp.maximum(r_b_diag, 1e-9),
    }


def policy_np(params: Params, q, b, r, v_o, tau, jac_ee, r_q_diag, r_b_diag):
    (_, grads) = value_and_grads_jit(params, q, b, r, v_o, tau)
    dv_dq, dv_db, dv_dr, _ = grads
    a_q = dv_dq + jnp.matmul(jnp.asarray(jac_ee, dtype=jnp.float32).T, dv_dr)
    a_b = dv_db + dv_dr
    u_q = -a_q / jnp.maximum(jnp.asarray(r_q_diag, dtype=jnp.float32), 1e-9)
    u_b = -a_b / jnp.maximum(jnp.asarray(r_b_diag, dtype=jnp.float32), 1e-9)
    return u_q, u_b
