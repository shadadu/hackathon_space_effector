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
    # print(f'shape of gra_q.T and grad_q: {(R_inv_diag*grad_q).shape}, {grad_q.shape}')
    # print(f'shape of grad_q.T and grad_q: {(grad_q*(R_inv_diag * grad_q)).shape}, {grad_q.shape}')
    # print(f'shape of grad_q.T and grad_q: {((grad_q * (R_inv_diag * grad_q)).sum(dim=-1)).shape}, {grad_q.shape}')
    # quad = (R_inv_diag * grad_q.T * grad_q).sum(dim=-1)  # (B,)
    # quad = (grad_q * grad_q.T).sum(dim=-1)  # (B,)
    quad = (grad_q * (R_inv_diag * grad_q)).sum(dim=-1)
    # quad = (R_inv_diag * torch.matmul(grad_q.T, grad_q)).sum(dim=-1)  # (B,)
    # print(f'shape of quad: {quad.shape}')
    # V_t : Neural Net quad: autograd (of Neural Net weights)
    # running_cost: hand-crafted; replace with Neural Net? For example: actor-critic in
    #               DeepPAAC and Aladi DGM Extension papers?

    resid = V_t + running_cost - 0.25 * quad
    return (resid ** 2).mean()

def terminal_loss(V_T, phi_T):
    """
    V_T: (B,)
    phi_T: (B,)
    """
    return ((V_T - phi_T) ** 2).mean()


import torch


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

    return torch.mean(residual ** 2)