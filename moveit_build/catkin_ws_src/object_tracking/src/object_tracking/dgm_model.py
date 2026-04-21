import torch
import torch.nn as nn

from typing import Any
import numpy as np


class DGMValueNet(nn.Module):
    """
    Position-only value function V(q, t, gpos):
      q: (7,)
      t: scalar normalized to [0,1]
      gpos: (3,) goal position in planning frame
    Input dim = 7 + 1 + 3 = 11
    Output = scalar V
    """

    def __init__(self, in_dim=11, hidden=256, depth=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DGMLayer(nn.Module):
    """
    Georgias Detorakis (2024): Practical Aspects on Solving Differential Equations Using Deep Learning: A Primer
    """

    def __init__(self, input_dim=1, hidden_size=50):
        super().__init__()

        self.Z_wg = nn.Linear(hidden_size, hidden_size)
        self.Z_ug = nn.Linear(input_dim, hidden_size, bias=False)

        self.G_wz = nn.Linear(hidden_size, hidden_size)
        self.G_uz = nn.Linear(input_dim, hidden_size, bias=False)

        self.R_wr = nn.Linear(hidden_size, hidden_size)
        self.R_ur = nn.Linear(input_dim, hidden_size, bias=False)

        self.H_wh = nn.Linear(hidden_size, hidden_size)
        self.H_uh = nn.Linear(input_dim, hidden_size, bias=False)

        # Non−linear Activation function
        self.sigma = nn.Tanh()

    def forward(self, x, s):
        Z = self.sigma(self.Z_wg(s) + self.Z_ug(x))
        # print(f"Z shape = {Z.shape}")
        G = self.sigma(self.G_wz(Z) + self.G_uz(x))
        # print(f"G shape = {G.shape}")
        R = self.sigma(self.R_wr(G) + self.R_ur(x))
        # print(f"R shape {R.shape} and s shape {s.shape} and I shape {self.H_wh(s).shape}")
        H = self.sigma(self.H_wh(s) * R + self.H_uh(x))
        # print(f"H shape {H.shape}")
        out = (1 - G) * H + Z * self.H_wh(s)
        # print(f"out shape {out.shape}")
        return out


class DGMLayer0(nn.Module):
    def __init__(self, input_dim=1, hidden_size=50):
        super().__init__()

        self.I_zu = nn.Linear(input_dim, hidden_size)
        self.dgm_layer = DGMLayer(input_dim, hidden_size)

    def forward(self, x, s):
        s1 = self.I_zu(s)
        # print(f"s1 shape {s1.shape}")
        out = self.dgm_layer(x, s1)
        # print(f"out shape {out.shape}")
        return out


class DGMLayerN(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_size=50):
        super().__init__()

        self.dgm_layer = DGMLayer(input_dim, hidden_size)
        self.K_zu = nn.Linear(hidden_size, output_dim)

    def forward(self, x, s):
        x1 = self.dgm_layer(x, s)
        # print(f"x1 shape {x1.shape}")
        out = self.K_zu(x1)
        # print(f"out shape {out.shape}")
        return out


class ValueNet(nn.Module):
    """
    dgm_layers = input_layer(DGMLayer0) + num_layers(hidden_dgm_layers) + output_layer (DGMLayerN)
    """

    def __init__(self, num_layers=1, input_dim=1, output_dim=1, hidden_size=50):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)

        self.layers = nn.ModuleList([DGMLayer0(input_dim, hidden_size)]) + \
                      nn.ModuleList([DGMLayer(input_dim, hidden_size) for _ in range(num_layers)]) + \
                      nn.ModuleList([DGMLayerN(input_dim, output_dim, hidden_size)])

    def forward(self, x, s):
        for i, layer in enumerate(self.layers):
            # print(f"layer {i} = {layer}")
            x = self.dropout(layer(s, x))

        # print(f"out shape: {x.shape}")
        return x.squeeze(-1)


class DGMLayer_(nn.Module):
    """
    Georgias Detorakis (2024): Practical Aspects on Solving Differential Equations Using Deep Learning: A Primer

    """

    def __init__(self, input_dim=1, hidden_size=50, expansion_factor=2):
        super().__init__()

        self.expanded_hidden_size = expansion_factor * hidden_size

        self.Z_wg = nn.Linear(self.expanded_hidden_size, self.expanded_hidden_size)
        self.Z_ug = nn.Linear(input_dim, self.expanded_hidden_size, bias=False)

        self.G_wz = nn.Linear(expansion_factor * hidden_size, expansion_factor * hidden_size)
        self.G_uz = nn.Linear(input_dim, self.expanded_hidden_size, bias=False)

        self.R_wr = nn.Linear(self.expanded_hidden_size, self.expanded_hidden_size)
        self.R_ur = nn.Linear(input_dim, self.expanded_hidden_size, bias=False)

        self.H_wh = nn.Linear(self.expanded_hidden_size, self.expanded_hidden_size)
        self.H_uh = nn.Linear(input_dim, self.expanded_hidden_size, bias=False)

        # Non−linear Activation function
        self.sigma = nn.Tanh()

    def forward(self, x, s):
        Z = self.sigma(self.Z_wg(s) + self.Z_ug(x))
        G = self.sigma(self.G_wz(Z) + self.G_uz(x))
        R = self.sigma(self.R_wr(G) + self.R_ur(x))
        H = self.sigma(self.H_wh(s) * R + self.H_uh(x))
        out = (1 - G) * H + Z * self.H_wh(s)
        return out


class DGMLayer0_(nn.Module):

    def __init__(self, input_dim=1, hidden_size=50, expansion_factor=2):
        super().__init__()

        self.I_zu = nn.Linear(input_dim, expansion_factor * hidden_size)
        self.dgm_layer = DGMLayer_(input_dim, hidden_size, expansion_factor=2)

    def forward(self, x, s):
        s1 = self.I_zu(s)
        out = self.dgm_layer(x, s1)
        return out


class DGMLayerN_(nn.Module):

    def __init__(self, input_dim=1, output_dim=1, hidden_size=192, expansion_factor=2):
        super().__init__()

        self.expanded_hidden_size = expansion_factor * hidden_size

        self.dgm_layer = DGMLayer_(input_dim, hidden_size, expansion_factor)
        self.dgm_layerN_ = nn.Linear(expansion_factor * hidden_size, hidden_size)
        self.K_zu = nn.Linear(hidden_size, output_dim)

    def forward(self, x, s):
        x1 = self.dgm_layer(x, s)
        x2 = self.dgm_layerN_(x1)
        out = self.K_zu(x2)
        return out


class ValueNet_(nn.Module):
    """
    num_dgm_layers
    """

    def __init__(self, num_layers=1, input_dim=1, output_dim=1, hidden_size=50, expansion_factor=2):
        super().__init__()

        self.dropout = nn.Dropout(p=0.1)

        self.layers = nn.ModuleList([DGMLayer0_(input_dim, hidden_size, expansion_factor)]) + \
                      nn.ModuleList([DGMLayer_(input_dim, hidden_size, expansion_factor) for _ in range(num_layers)]) + \
                      nn.ModuleList([DGMLayerN_(input_dim, output_dim, hidden_size, expansion_factor)])

    #         self.dgm_layer = DGMLayer(input_dim, hidden_size)

    def forward(self, x, s):
        for i, layer in enumerate(self.layers):
            x = self.dropout(layer(s, x))
        return x.squeeze(-1)


def build_input(q, t_norm, gpos):
    # q: (B,7), t_norm: (B,1), gpos: (B,3)
    return torch.cat([q, t_norm, gpos], dim=-1)


def save_checkpoint(path, model, meta: dict):
    ckpt = {
        "state_dict": model.state_dict(),
        "meta": meta,
    }
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    meta = ckpt.get("meta", {})
    model = DGMValueNet(
        in_dim=meta.get("in_dim", 11),
        hidden=meta.get("hidden", 256),
        depth=meta.get("depth", 4),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, meta
