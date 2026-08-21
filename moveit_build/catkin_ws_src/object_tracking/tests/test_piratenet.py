import pickle

import jax
import jax.numpy as jnp
import numpy as np

from object_tracking.dgm_jax import (
    adam_init,
    apply_value,
    fit_piratenet_output_lstsq,
    init_mlp_params,
    init_piratenet_params,
    load_checkpoint,
    piratenet_features,
    save_checkpoint,
)


def test_zero_alpha_makes_every_block_an_identity():
    params = init_piratenet_params(jax.random.PRNGKey(0), in_dim=3, hidden=8, blocks=3)
    x = jnp.array([[0.2, -0.4, 0.7]], dtype=jnp.float32)
    phase = x @ params["fourier"].T
    embedding = jnp.concatenate([jnp.cos(phase), jnp.sin(phase)], axis=-1)
    np.testing.assert_allclose(piratenet_features(params, x), embedding, rtol=1e-6, atol=1e-6)


def test_value_gradient_is_finite_in_physical_coordinates():
    params = init_piratenet_params(
        jax.random.PRNGKey(1), in_dim=3, hidden=8, blocks=2,
        input_min=[-2.0, 0.0, 10.0], input_max=[2.0, 4.0, 20.0],
    )
    grad = jax.grad(lambda x: apply_value(params, x[None, :])[0])(
        jnp.array([0.1, 2.0, 12.0], dtype=jnp.float32)
    )
    assert grad.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(grad)))


def test_physics_informed_output_fit_reconstructs_boundary_values():
    params = init_piratenet_params(jax.random.PRNGKey(2), in_dim=1, hidden=8, blocks=2)
    x = jnp.linspace(-0.9, 0.9, 32)[:, None]
    targets = 0.5 + jnp.sin(x[:, 0])
    fitted = fit_piratenet_output_lstsq(params, x, targets, ridge=1e-5)
    error = jnp.mean((apply_value(fitted, x) - targets) ** 2)
    assert float(error) < 1e-4


def test_checkpoint_round_trip_and_legacy_loading(tmp_path):
    pirate = init_piratenet_params(jax.random.PRNGKey(3), in_dim=3, hidden=8, blocks=2)
    path = tmp_path / "pirate.pkl"
    save_checkpoint(path, pirate, adam_init(pirate), {"architecture": "piratenet"}, 1.0)
    restored, meta = load_checkpoint(path)
    assert meta["architecture"] == "piratenet"
    np.testing.assert_allclose(apply_value(restored, jnp.ones((1, 3))), apply_value(pirate, jnp.ones((1, 3))))

    legacy = init_mlp_params(jax.random.PRNGKey(4), in_dim=3, hidden=4, depth=2)
    legacy_path = tmp_path / "legacy.pkl"
    save_checkpoint(legacy_path, legacy, adam_init(legacy), {}, 2.0)
    with legacy_path.open("rb") as handle:
        assert pickle.load(handle)["format"] == "dgm_jax_mlp_v1"
    restored_legacy, _ = load_checkpoint(legacy_path)
    np.testing.assert_allclose(apply_value(restored_legacy, jnp.ones((1, 3))), apply_value(legacy, jnp.ones((1, 3))))
