import torch
import torch.nn as nn


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
