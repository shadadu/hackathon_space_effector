from typing import Any

import torch
import torch.nn as nn

class DGMLayer(nn.Module):
    """
    Georgias Detorakis (2024): Practical Aspects on Solving Differential Equations Using Deep Learning: A Primer

    """

    def __init__(self, input_dim=1, hidden_size=50):
        super().__init__()

        self.I_init = nn.Linear(input_dim, hidden_size)
        self.Z_wg = nn.Linear(hidden_size, hidden_size)
        self.Z_ug = nn.Linear(input_dim, hidden_size, bias=False)

        self.G_wz = nn.Linear(hidden_size, hidden_size)
        self.G_uz = nn.Linear(input_dim, hidden_size, bias=False)

        self.R_wr = nn.Linear(hidden_size, hidden_size)
        self.R_ur = nn.Linear(input_dim, hidden_size, bias=False)

        self.H_wh = nn.Linear(hidden_size, hidden_size)
        self.H_uh = nn.Linear(input_dim, hidden_size, bias=False)

        # Non−linear Activation funcdtion
        self.sigma = nn.Tanh()

    def forward(self, x, s):
        I = self.I_init(s)

        Z = self.sigma(self.Z_wg(s) + self.Z_ug(x))
        G = self.sigma(self.G_wz(s) + self.G_uz(x))
        R = self.sigma(self.R_wr(s) + self.R_ur(x))
        print(f"s")
        H = self.sigma(self.H_wh(s * R) + self.H_uh(x))
        out = (1-G)*H + Z*s
        # out = torch.sub(1, G) * H + Z * s
        return out



class DGMValueNet(nn.Module):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(args, kwargs)

    def init(self, num_layers, input_dim, hidden_size):
        super().__init__()



    pass