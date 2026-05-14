import torch


def hjb_residual_loss(V, q, t_norm, running_cost, R_inv_diag):
    """
    V: (B,) value output
    q: (B,7) requires_grad=True
    t_norm: (B,1) requires_grad=True
    running_cost: (B,) (already computed)
    R_inv_diag: (7,) tensor

    Residual: V_t + l(q,t) - 1/4 * grad_q^T R^{-1} grad_q = 0
    """
    grad_q = torch.autograd.grad(V.sum(), q, create_graph=True, retain_graph=True)[0]  # (B,7)
    V_t = torch.autograd.grad(V.sum(), t_norm, create_graph=True, retain_graph=True)[0].squeeze(-1)  # (B,)
    quad = (grad_q * (R_inv_diag * grad_q)).sum(dim=-1)
    # V_t : Neural Net quad: autograd (of Neural Net weights)
    # running_cost: hand-crafted; replace with Neural Net? For example: actor-critic in
    #               DeepPAAC and Aladi DGM Extension papers?

    resid = V_t + running_cost - 0.25 * quad
    control_velocity = - 0.25 * quad
    return (resid ** 2).mean(), control_velocity


# def terminal_loss(V_T, phi_T):
#     """
#     V_T: (B,)
#     phi_T: (B,)
#     """
#     return ((V_T - phi_T) ** 2).mean()


def dist_term(value, min_limit, max_limit):
    """
    value: tensor or scalar
    Returns squared distance to nearest bound.
    """
    d = torch.minimum(value - min_limit, max_limit - value)
    return d ** 2


def terminal_loss(VT, phi):
    VT = VT.reshape(-1)
    phi = phi.reshape(-1)
    return torch.mean((VT - phi) ** 2)


def terminal_position_cost(phi):
    phi = phi.reshape(-1)
    return (torch.mean(phi)) ** 2


def initial_condition_cost(item):
    item = item.reshape(-1)
    return (torch.mean(item)) ** 2


def time_monotonicity_cost(t):
    s = 0
    i = 0
    for _ in range(0, len(t) - 1):
        diff = t[i + 1] - t[i]
        if diff >= 0:
            s += 0
        else:
            s += 1
        i += 1
    return s


def hjb_residual_loss_(V, q, t, running_cost, R_inv_diag, vel_limits, Cv=0.0, T=1.0):
    """
    Example HJB residual for dynamics:
        q_dot = u

    and Hamiltonian:
        H(q, grad V) = l(q,u) + gradV^T u
    with optimal control:
        u* = -0.5 * R^{-1} * grad_q V

    We add:
        Cv * joint_velocity_limit_penalty(u*)
    to the running cost.

    Inputs
    ------
    V            : (B,1) or (B,)
    q            : (B,7), requires_grad=True
    t            : (B,1), requires_grad=True, normalized time
    running_cost : (B,)
    R_inv_diag   : (7,)
    vel_limits   : (7,)
    Cv           : scalar
    """

    if V.dim() == 2 and V.shape[1] == 1:
        V_scalar = V[:, 0]
    else:
        V_scalar = V.reshape(-1)

    dV_dq = torch.autograd.grad(
        outputs=V_scalar.sum(),
        inputs=q,
        create_graph=True
    )[0]  # (B,7)

    dV_dt = torch.autograd.grad(
        outputs=V_scalar.sum(),
        inputs=t,
        create_graph=True
    )[0].reshape(-1)  # (B,)

    # Optimal control u* = -0.5 * R^{-1} * grad_q V
    # broadcast (7,) across batch
    u_star = -0.5 * dV_dq * R_inv_diag.view(1, -1)  # (B,7)

    # Quadratic control effort term: u^T R u
    # Since R_inv_diag = 1/R_diag, then R_diag = 1/R_inv_diag
    R_diag = 1.0 / R_inv_diag
    control_cost = torch.sum((u_star ** 2) * R_diag.view(1, -1), dim=1)  # (B,)

    # bounds are [-vel_limit, +vel_limit]
    vel_penalty_terms = []
    for j in range(u_star.shape[1]):
        d2 = dist_term(
            u_star[:, j],
            -vel_limits[j],
            vel_limits[j]
        )
        vel_penalty_terms.append(1.0 / (d2 + 1e-6))

    joint_velocity_limit_penalty = torch.stack(vel_penalty_terms, dim=1).sum(dim=1)

    total_running_cost = running_cost + control_cost + Cv * joint_velocity_limit_penalty

    # gradV · f with f=u*
    drift_term = torch.sum(dV_dq * u_star, dim=1)

    # HJB PDE:
    # dV/dt + l(q,u*) + gradV·u* = 0
    residual = dV_dt + total_running_cost + drift_term

    return torch.mean(residual ** 2), u_star


def hjb_residual_(
        V,
        q,
        t_norm,
        running_cost,
        R_inv_diag,
        reduction="mean",
        return_residual=False,
):
    """
    HJB residual loss for kinematic optimal control.

    Assumes control cost:

        0.5 * u^T R u

    and optimal control:

        u* = -R^{-1} grad_q V

    HJB residual:

        V_t + running_cost - 0.5 * grad_q V^T R^{-1} grad_q V = 0

    Args:
        V:
            Value function output, shape (batch, 1) or (batch,).

        q:
            Joint state tensor, shape (batch, 7), requires_grad=True.

        t_norm:
            Normalized time tensor, shape (batch, 1), requires_grad=True.
            If t_norm = t / T, then V_t here is with respect to normalized time.

        running_cost:
            Per-sample running cost, shape (batch,) or (batch, 1).

        R_inv_diag:
            Diagonal inverse control-cost weights, shape (7,).

        reduction:
            "mean", "sum", or "none".

        return_residual:
            If True, return (loss, residual_vec).

    Returns:
        loss, or (loss, residual_vec)
    """

    if V.ndim == 2 and V.shape[1] == 1:
        V_scalar = V[:, 0]
    else:
        V_scalar = V.reshape(-1)

    running_cost = running_cost.reshape(-1)

    if R_inv_diag.device != q.device:
        R_inv_diag = R_inv_diag.to(q.device)

    if R_inv_diag.dtype != q.dtype:
        R_inv_diag = R_inv_diag.to(dtype=q.dtype)

    grad_q = torch.autograd.grad(
        outputs=V_scalar.sum(),
        inputs=q,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_t = torch.autograd.grad(
        outputs=V_scalar.sum(),
        inputs=t_norm,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    V_t = grad_t.reshape(-1)

    quadratic_control_term = 0.5 * torch.sum(
        grad_q * grad_q * R_inv_diag.view(1, -1),
        dim=1,
    )

    residual_vec = V_t + running_cost - quadratic_control_term

    residual_sq = residual_vec ** 2

    if reduction == "mean":
        loss = residual_sq.mean()
    elif reduction == "sum":
        loss = residual_sq.sum()
    elif reduction == "none":
        loss = residual_sq
    else:
        raise ValueError(
            f"Unknown reduction='{reduction}'. Use 'mean', 'sum', or 'none'."
        )

    if return_residual:
        return loss, residual_vec, -quadratic_control_term

    return loss, -quadratic_control_term


def hjb_residual(
        V,
        q,
        t_norm,
        running_cost,
        R_inv_diag,
        reduction="mean",
        return_residual=False,
):
    """
    HJB residual loss for kinematic optimal control.

    Assumes control cost:

        0.5 * u^T R u

    and optimal control:

        u* = -R^{-1} grad_q V

    HJB residual:

        V_t + running_cost - 0.5 * grad_q V^T R^{-1} grad_q V = 0

    Args:
        V:
            Value function output, shape (batch, 1) or (batch,).

        q:
            Joint state tensor, shape (batch, 7), requires_grad=True.

        t_norm:
            Normalized time tensor, shape (batch, 1), requires_grad=True.
            If t_norm = t / T, then V_t here is with respect to normalized time.

        running_cost:
            Per-sample running cost, shape (batch,) or (batch, 1).

        R_inv_diag:
            Diagonal inverse control-cost weights, shape (7,).

        reduction:
            "mean", "sum", or "none".

        return_residual:
            If True, return (loss, residual_vec).

    Returns:
        loss, or (loss, residual_vec)
    """

    if V.ndim == 2 and V.shape[1] == 1:
        V_scalar = V[:, 0]
    else:
        V_scalar = V.reshape(-1)

    running_cost = running_cost.reshape(-1)

    if R_inv_diag.device != q.device:
        R_inv_diag = R_inv_diag.to(q.device)

    if R_inv_diag.dtype != q.dtype:
        R_inv_diag = R_inv_diag.to(dtype=q.dtype)

    grad_q = torch.autograd.grad(
        outputs=V_scalar.sum(),
        inputs=q,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    grad_t = torch.autograd.grad(
        outputs=V_scalar.sum(),
        inputs=t_norm,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    V_t = grad_t.reshape(-1)

    quadratic_control_term = 0.5 * torch.sum(
        grad_q * grad_q * R_inv_diag.view(1, -1),
        dim=1,
    )

    residual_vec = V_t + running_cost - quadratic_control_term

    residual_sq = residual_vec ** 2

    if reduction == "mean":
        loss = residual_sq.mean()
    elif reduction == "sum":
        loss = residual_sq.sum()
    elif reduction == "none":
        loss = residual_sq
    else:
        raise ValueError(
            f"Unknown reduction='{reduction}'. Use 'mean', 'sum', or 'none'."
        )

    u_ic = -torch.mean((0.5 * grad_q * grad_q * R_inv_diag.view(1, -1))[0:3], dim=0)
    u_tc = -torch.mean((0.5 * grad_q * grad_q * R_inv_diag.view(1, -1))[:-8], dim=0)

    if return_residual:
        return loss, residual_vec, u_ic, u_tc

    return loss, u_ic, u_tc
