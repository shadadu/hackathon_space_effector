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

    quad = (grad_q * R_inv_diag[None, :] * grad_q).sum(dim=-1)  # (B,)
    resid = V_t + running_cost - 0.25 * quad
    return (resid ** 2).mean()


def terminal_loss(V_T, phi_T):
    """
    V_T: (B,)
    phi_T: (B,)
    """
    return ((V_T - phi_T) ** 2).mean()