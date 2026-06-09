import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np


Params = List[Dict[str, jnp.ndarray]]
AdamState = Dict[str, Any]


def build_input(q, t_norm, gpos):
    return jnp.concatenate([q, t_norm, gpos], axis=-1)


def init_mlp_params(key, in_dim=11, hidden=256, depth=4) -> Params:
    dims = [in_dim] + [hidden] * depth + [1]
    keys = jax.random.split(key, len(dims) - 1)
    params = []
    for k, din, dout in zip(keys, dims[:-1], dims[1:]):
        limit = jnp.sqrt(6.0 / float(din + dout))
        params.append({
            "w": jax.random.uniform(k, (din, dout), minval=-limit, maxval=limit, dtype=jnp.float32),
            "b": jnp.zeros((dout,), dtype=jnp.float32),
        })
    return params


def apply_mlp(params: Params, x):
    y = x
    for layer in params[:-1]:
        y = jnp.tanh(jnp.matmul(y, layer["w"]) + layer["b"])
    y = jnp.matmul(y, params[-1]["w"]) + params[-1]["b"]
    return jnp.squeeze(y, axis=-1)


def value_scalar(params: Params, q, t_norm, gpos):
    q = jnp.asarray(q, dtype=jnp.float32).reshape(1, 7)
    t_norm = jnp.asarray(t_norm, dtype=jnp.float32).reshape(1, 1)
    gpos = jnp.asarray(gpos, dtype=jnp.float32).reshape(1, 3)
    return apply_mlp(params, build_input(q, t_norm, gpos))[0]


value_grad_q = jax.jit(jax.grad(value_scalar, argnums=1))


def policy_grad_q_np(params: Params, q, t_norm, gpos):
    grad_q = value_grad_q(
        params,
        jnp.asarray(q, dtype=jnp.float32),
        jnp.asarray([t_norm], dtype=jnp.float32),
        jnp.asarray(gpos, dtype=jnp.float32),
    )
    return np.asarray(grad_q, dtype=np.float64).reshape(7)


def terminal_loss(params: Params, q_t, t_t, g_t, phi_t):
    v_t = apply_mlp(params, build_input(q_t, t_t, g_t))
    return jnp.mean((v_t.reshape(-1) - phi_t.reshape(-1)) ** 2)


def hjb_residual_loss(params: Params, q, t_norm, gpos, running_cost, r_inv_diag):
    def v_for_q_t(q_i, t_i, g_i):
        return value_scalar(params, q_i, t_i, g_i)

    d_v_dq = jax.vmap(jax.grad(v_for_q_t, argnums=0))(q, t_norm, gpos)
    d_v_dt = jax.vmap(jax.grad(v_for_q_t, argnums=1))(q, t_norm, gpos).reshape(-1)
    quad = jnp.sum(d_v_dq * (r_inv_diag.reshape(1, -1) * d_v_dq), axis=-1)
    residual = d_v_dt + running_cost.reshape(-1) - 0.25 * quad
    return jnp.mean(residual ** 2)


def total_loss(params: Params, batch: Dict[str, jnp.ndarray], cpd: float, ctr: float):
    loss_pde = hjb_residual_loss(
        params,
        batch["q"],
        batch["t"],
        batch["g"],
        batch["running_cost"],
        batch["r_inv_diag"],
    )
    loss_term = terminal_loss(
        params,
        batch["q_t"],
        batch["t_t"],
        batch["g_t"],
        batch["phi_t"],
    )
    return cpd * loss_pde + ctr * loss_term, (loss_pde, loss_term)


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
def train_step(params: Params, opt_state: AdamState, batch: Dict[str, jnp.ndarray], lr: float, cpd: float, ctr: float):
    (loss, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(params, batch, cpd, ctr)
    params, opt_state = adam_update(params, grads, opt_state, lr)
    return params, opt_state, loss, aux[0], aux[1]


def make_batch(q_np, t_np, g_np, running_cost_np, q_t_np, t_t_np, g_t_np, phi_t_np, r_inv_diag_np):
    return {
        "q": jnp.asarray(q_np, dtype=jnp.float32),
        "t": jnp.asarray(t_np, dtype=jnp.float32),
        "g": jnp.asarray(g_np, dtype=jnp.float32),
        "running_cost": jnp.asarray(running_cost_np, dtype=jnp.float32),
        "q_t": jnp.asarray(q_t_np, dtype=jnp.float32),
        "t_t": jnp.asarray(t_t_np, dtype=jnp.float32),
        "g_t": jnp.asarray(g_t_np, dtype=jnp.float32),
        "phi_t": jnp.asarray(phi_t_np, dtype=jnp.float32),
        "r_inv_diag": jnp.asarray(r_inv_diag_np, dtype=jnp.float32),
    }


def params_to_numpy(params: Params):
    return jax.tree_util.tree_map(lambda x: np.asarray(x), params)


def params_from_numpy(params_np):
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float32), params_np)


def save_checkpoint(path, params: Params, opt_state: AdamState, meta: Dict[str, Any], loss: float):
    ckpt = {
        "format": "dgm_jax_mlp_v1",
        "params": params_to_numpy(params),
        "opt_state": params_to_numpy(opt_state),
        "meta": dict(meta),
        "loss": float(loss),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(ckpt, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(path) -> Tuple[Params, Dict[str, Any]]:
    with Path(path).open("rb") as f:
        ckpt = pickle.load(f)
    if ckpt.get("format") != "dgm_jax_mlp_v1":
        raise ValueError(f"Unsupported DGM JAX checkpoint format: {ckpt.get('format')}")
    return params_from_numpy(ckpt["params"]), ckpt.get("meta", {})
