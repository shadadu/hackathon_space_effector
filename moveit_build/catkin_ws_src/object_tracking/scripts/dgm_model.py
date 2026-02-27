#!/usr/bin/env python3
import torch
import torch.nn as nn


class DGMValueNet(nn.Module):
    """
    Simple MLP value function approximator V(q, t, g).
    V: R^(7 + 1 + 3) -> R
      q: 7 joint angles (active arm joints)
      t: scalar time in [0, T]
      g: goal position in planning frame (x,y,z)

    Position-only v1.
    """

    def __init__(self, in_dim: int = 11, hidden: int = 256, depth: int = 4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.Tanh()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, in_dim)
        return self.net(x).squeeze(-1)  # (B,)


    def build_input(q: torch.Tensor, t: torch.Tensor, gpos: torch.Tensor) -> torch.Tensor:
        """
        q:   (B,7)
        t:   (B,1)
        gpos:(B,3)
        """
        return torch.cat([q, t, gpos], dim=-1)


    def load_model(path: str, device: str = "cpu") -> DGMValueNet:
        ckpt = torch.load(path, map_location=device)
        model = DGMValueNet(
            in_dim=ckpt.get("in_dim", 11),
            hidden=ckpt.get("hidden", 256),
            depth=ckpt.get("depth", 4),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
