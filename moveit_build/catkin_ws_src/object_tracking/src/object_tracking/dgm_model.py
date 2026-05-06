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

    def __init__(self, input_dim, hidden_dim):
        super(DGMLayer, self).__init__()
        # Standard fully connected layers used for the internal gates
        self.Uz = nn.Linear(input_dim, hidden_dim)
        self.Ug = nn.Linear(input_dim, hidden_dim)
        self.Ur = nn.Linear(input_dim, hidden_dim)
        self.Uh = nn.Linear(input_dim, hidden_dim)

        self.Wz = nn.Linear(hidden_dim, hidden_dim)
        self.Wg = nn.Linear(hidden_dim, hidden_dim)
        self.Wr = nn.Linear(hidden_dim, hidden_dim)
        self.Wh = nn.Linear(hidden_dim, hidden_dim)

        self.activation = nn.Tanh()

    def forward(self, x, S):
        # x is the input coordinates (spatial/temporal)
        # S is the output of the previous layer
        z = self.activation(self.Uz(x) + self.Wz(S))
        g = self.activation(self.Ug(x) + self.Wg(S))
        r = self.activation(self.Ur(x) + self.Wr(S))
        h = self.activation(self.Uh(x) + self.Wh(S * r))

        # Element-wise gate update
        return (1 - g) * h + z * S


class DGMLayer0(nn.Module):
    def __init__(self, input_dim=1, hidden_size=50):
        super().__init__()

        self.I_zu = nn.Linear(input_dim, hidden_size)
        self.dgm_layer = DGMLayer(input_dim, hidden_size)

    def forward(self, x, s):
        s1 = self.I_zu(s)
        out = self.dgm_layer(x, s1)
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

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1):
        super(ValueNet, self).__init__()
        self.initial_layer = nn.Linear(input_dim, hidden_dim)
        self.dgm_layers = nn.ModuleList([
            DGMLayer(input_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initial transformation
        S = torch.tanh(self.initial_layer(x))

        # Pass through specialized DGM layers
        for layer in self.dgm_layers:
            S = layer(x, S)

        return self.final_layer(S)


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


class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.dgm = DGMValueNet(in_dim=48, hidden=192, depth=24)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Shortcut to align dimensions for the skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        x = self.dgm(x)
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class ResNet1D(nn.Module):
    def __init__(self, input_channels=11, out_channels=64, num_layers=16, num_classes=10, dim=16):
        super().__init__()
        # Initial stem: reduces temporal length from 192 -> 48
        self.prep = nn.Sequential(
            nn.Conv1d(input_channels, out_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.ln = nn.Conv1d(11, 256, kernel_size=1, stride=1, padding=1, bias=False)

        self.layers = nn.ModuleList([ResBlock1D(out_channels, out_channels) for _ in range(num_layers)])

        # Residual Layers
        self.layer1 = ResBlock1D(out_channels, out_channels)
        self.layer2 = ResBlock1D(out_channels, 128, stride=2)  # Length 48 -> 24

        # Output head
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        a, b = x.shape

        y = nn.Conv1d(a, 192, kernel_size=3, stride=1, padding=1, bias=False)(x)
        c, d = y.shape
        x = y.T.reshape([1, d, c])

        x = self.prep(x)

        for i, layer in enumerate(self.layers):
            x = layer(x)

        x = self.layer2(x)
        x = self.avgpool(x).squeeze(-1)
        x = nn.Linear(128, a)(x)
        return x


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
