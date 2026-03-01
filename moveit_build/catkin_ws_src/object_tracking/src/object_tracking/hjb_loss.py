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
    # print(f'Shapes of tensors, value of R_inv_diag: {grad_q.shape}, {grad_q.T.shape}, {R_inv_diag.shape}, {R_inv_diag}')
    # quad = (grad_q.T * R_inv_diag * grad_q).sum(dim=-1)  # (B,)
    quad = R_inv_diag * torch.matmul(grad_q, grad_q.T).sum(dim=-1)  # (B,)
    # print(f'shape of quad: {torch.matmul(grad_q.T, grad_q).shape}, {quad.shape}')
    resid = V_t + running_cost - 0.25 * quad
    return (resid ** 2).mean()


def terminal_loss(V_T, phi_T):
    """
    V_T: (B,)
    phi_T: (B,)
    """
    return ((V_T - phi_T) ** 2).mean()