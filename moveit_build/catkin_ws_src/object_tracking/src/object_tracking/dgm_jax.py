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


def init_piratenet_params(
        key,
        in_dim=11,
        hidden=256,
        blocks=4,
        fourier_scale=1.0,
        input_min=None,
        input_max=None,
):
    """Initialize a PirateNet whose residual blocks are identities at step zero."""
    if hidden <= 0 or hidden % 2:
        raise ValueError("PirateNet hidden size must be a positive even integer")
    if blocks <= 0:
        raise ValueError("PirateNet must contain at least one residual block")

    input_min = -jnp.ones((in_dim,), dtype=jnp.float32) if input_min is None else jnp.asarray(input_min, dtype=jnp.float32)
    input_max = jnp.ones((in_dim,), dtype=jnp.float32) if input_max is None else jnp.asarray(input_max, dtype=jnp.float32)
    if input_min.shape != (in_dim,) or input_max.shape != (in_dim,):
        raise ValueError(f"Expected input bounds with shape ({in_dim},)")
    if np.any(np.asarray(input_max) <= np.asarray(input_min)):
        raise ValueError("Each PirateNet input maximum must exceed its minimum")

    def dense(k, din, dout):
        limit = jnp.sqrt(6.0 / float(din + dout))
        return {
            "w": jax.random.uniform(k, (din, dout), minval=-limit, maxval=limit, dtype=jnp.float32),
            "b": jnp.zeros((dout,), dtype=jnp.float32),
        }

    keys = iter(jax.random.split(key, 4 + 3 * blocks))
    params = {
        "fourier": jax.random.normal(next(keys), (hidden // 2, in_dim), dtype=jnp.float32) * float(fourier_scale),
        "input_min": input_min,
        "input_max": input_max,
        "u": dense(next(keys), hidden, hidden),
        "v": dense(next(keys), hidden, hidden),
        "blocks": [],
    }
    for _ in range(blocks):
        params["blocks"].append({
            "layers": [dense(next(keys), hidden, hidden) for _ in range(3)],
            "alpha": jnp.array(0.0, dtype=jnp.float32),
        })
    params["output"] = dense(next(keys), hidden, 1)
    return params


def apply_mlp(params: Params, x):
    y = x
    for layer in params[:-1]:
        y = jnp.tanh(jnp.matmul(y, layer["w"]) + layer["b"])
    y = jnp.matmul(y, params[-1]["w"]) + params[-1]["b"]
    return jnp.squeeze(y, axis=-1)


def _dense(layer, x):
    return jnp.matmul(x, layer["w"]) + layer["b"]


def piratenet_features(params, x):
    """Return the features consumed by the PirateNet scalar output layer."""
    x = jnp.asarray(x, dtype=jnp.float32)
    x_min = jax.lax.stop_gradient(params["input_min"])
    x_max = jax.lax.stop_gradient(params["input_max"])
    x_norm = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    fourier = jax.lax.stop_gradient(params["fourier"])
    phase = jnp.matmul(x_norm, fourier.T)
    features = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    u = jnp.tanh(_dense(params["u"], features))
    v = jnp.tanh(_dense(params["v"], features))
    state = features
    for block in params["blocks"]:
        f = jnp.tanh(_dense(block["layers"][0], state))
        z1 = f * u + (1.0 - f) * v
        g = jnp.tanh(_dense(block["layers"][1], z1))
        z2 = g * u + (1.0 - g) * v
        h = jnp.tanh(_dense(block["layers"][2], z2))
        state = block["alpha"] * h + (1.0 - block["alpha"]) * state
    return state


def apply_value(params, x):
    if isinstance(params, dict) and "blocks" in params and "fourier" in params:
        y = _dense(params["output"], piratenet_features(params, x))
        return jnp.squeeze(y, axis=-1)
    return apply_mlp(params, x)


def fit_piratenet_output_lstsq(params, x, targets, weights=None, ridge=1e-6):
    """Fit the initial linear output to known boundary values."""
    if not (isinstance(params, dict) and "blocks" in params):
        raise ValueError("Physics-informed output initialization requires PirateNet parameters")
    features = piratenet_features(params, x)
    design = jnp.concatenate([features, jnp.ones((features.shape[0], 1), dtype=features.dtype)], axis=1)
    targets = jnp.asarray(targets, dtype=features.dtype).reshape(-1, 1)
    if weights is not None:
        scale = jnp.sqrt(jnp.asarray(weights, dtype=features.dtype).reshape(-1, 1))
        design = design * scale
        targets = targets * scale
    gram = design.T @ design + ridge * jnp.eye(design.shape[1], dtype=design.dtype)
    solution = jnp.linalg.solve(gram, design.T @ targets)
    updated = dict(params)
    updated["output"] = {"w": solution[:-1], "b": solution[-1].reshape(1)}
    return updated


def value_scalar(params: Params, q, t_norm, gpos):
    q = jnp.asarray(q, dtype=jnp.float32).reshape(1, 7)
    t_norm = jnp.asarray(t_norm, dtype=jnp.float32).reshape(1, 1)
    gpos = jnp.asarray(gpos, dtype=jnp.float32).reshape(1, 3)
    return apply_value(params, build_input(q, t_norm, gpos))[0]


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
    v_t = apply_value(params, build_input(q_t, t_t, g_t))
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
    architecture = "piratenet" if isinstance(params, dict) and "blocks" in params else "mlp"
    ckpt = {
        "format": "value_net_jax_v2" if architecture == "piratenet" else "dgm_jax_mlp_v1",
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
    if ckpt.get("format") not in ("dgm_jax_mlp_v1", "value_net_jax_v2"):
        raise ValueError(f"Unsupported DGM JAX checkpoint format: {ckpt.get('format')}")
    return params_from_numpy(ckpt["params"]), ckpt.get("meta", {})
