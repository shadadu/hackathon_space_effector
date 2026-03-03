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

    # def load_model(path: str, device: str = "cpu") -> DGMValueNet:
    #     ckpt = torch.load(path, map_location=device)
    #
    #     # Case 1: full model was saved
    #     if isinstance(ckpt, DGMValueNet):
    #         model = ckpt
    #
    #     # Case 2: checkpoint dict
    #     elif isinstance(ckpt, dict):
    #
    #         # Extract state_dict
    #         state_dict = ckpt.get("state_dict", ckpt)
    #
    #         # Infer architecture safely
    #         in_dim = ckpt.get("in_dim")
    #         hidden = ckpt.get("hidden")
    #         depth = ckpt.get("depth")
    #
    #         if in_dim is None or hidden is None or depth is None:
    #             raise ValueError(
    #                 "Checkpoint missing architecture parameters "
    #                 "(in_dim, hidden, depth). "
    #                 "You must save them during training."
    #             )
    #
    #         model = ModelOps.DGMValueNet(
    #             in_dim=in_dim,
    #             hidden=hidden,
    #             depth=depth,
    #         )
    #
    #         # Handle DataParallel
    #         if list(state_dict.keys())[0].startswith("module."):
    #             state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #
    #         model.load_state_dict(state_dict)
    #
    #     else:
    #         raise TypeError("Unknown checkpoint format")
    #
    #     model.to(device)
    #     model.eval()
    #     return model





