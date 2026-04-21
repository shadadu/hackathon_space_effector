#!/usr/bin/env python3
import os
import time
from typing import List, Tuple
import numpy as np
import torch
import torch.optim as optim

import rospy
from geometry_msgs.msg import Point
from moveit_msgs.srv import GetPositionFK, GetPositionFKRequest
from moveit_commander import RobotCommander, MoveGroupCommander

from object_tracking.dgm_model import DGMValueNet, build_input, save_checkpoint
from object_tracking.hjb_loss import hjb_residual_loss, terminal_loss
from object_tracking.fk_client import FKClient


def panda_joint_limits() -> Tuple[np.ndarray, np.ndarray]:
    # Hard-coded Panda joint limits (rad) for safety/stability.
    # Matches common Panda model limits.
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973], dtype=np.float64)
    return jmin, jmax


def sample_goal_positions(n: int) -> np.ndarray:
    # Simple reachable-ish workspace box in panda_link0/world vicinity
    # You can tighten later once you confirm real reachability.
    xs = np.random.uniform(0.25, 0.65, size=(n, 1))
    ys = np.random.uniform(-0.30, 0.30, size=(n, 1))
    zs = np.random.uniform(0.10, 0.60, size=(n, 1))
    return np.hstack([xs, ys, zs]).astype(np.float64)


class FKHelper:
    def __init__(self, fk_service="/compute_fk", ee_link="panda_hand", frame="world"):
        rospy.wait_for_service(fk_service, timeout=30.0)
        self.fk = rospy.ServiceProxy(fk_service, GetPositionFK)
        self.ee_link = ee_link
        self.frame = frame

    def fk_pos(self, joint_names: List[str], q: np.ndarray) -> np.ndarray:
        req = GetPositionFKRequest()
        req.fk_link_names = [self.ee_link]
        req.header.frame_id = self.frame
        req.robot_state.joint_state.name = list(joint_names)
        req.robot_state.joint_state.position = q.tolist()
        resp = self.fk(req)
        if resp.error_code.val != 1 or not resp.pose_stamped:
            raise RuntimeError(f"FK failed code={resp.error_code.val}")
        p = resp.pose_stamped[0].pose.position
        return np.array([p.x, p.y, p.z], dtype=np.float64)


def main():
    rospy.init_node("dgm_train", anonymous=True)

    # Hyperparams (match your request)
    T = float(rospy.get_param("~T", 2.0))
    dt = float(rospy.get_param("~dt", 0.02))  # used for normalization + later rollout compatibility
    device = rospy.get_param("~device", "cpu")

    # Training params
    iters = int(rospy.get_param("~iters", 4000))
    batch = int(rospy.get_param("~batch", 256))
    lr = float(rospy.get_param("~lr", 3e-4))
    depth = int(rospy.get_param("~depth", 4))
    hidden = int(rospy.get_param("~hidden", 256))

    # Cost weights (position-only)
    Qp = float(rospy.get_param("~Qp", 10.0))         # running position error weight
    QpT = float(rospy.get_param("~Qp_terminal", 80.0))  # terminal weight

    # Control cost R (diagonal)
    R_diag = np.array(rospy.get_param("~R_diag", [0.15]*7), dtype=np.float64)  # slightly penalize joint speed
    R_inv = 1.0 / np.maximum(R_diag, 1e-9)

    # Model save
    out_path = rospy.get_param("~out_path", "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pth")

    robot = RobotCommander()
    group = MoveGroupCommander("panda_arm")
    joint_names = group.get_active_joints()
    if len(joint_names) != 7:
        raise RuntimeError(f"Expected 7 active joints for panda_arm, got {len(joint_names)}: {joint_names}")

    # FK service
    fk_svc = rospy.get_param("~fk_service", "/compute_fk")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    frame = rospy.get_param("~frame", "world")
    fk = FKHelper(fk_service=fk_svc, ee_link=ee_link, frame=frame)

    jmin, jmax = panda_joint_limits()

    model = DGMValueNet(in_dim=11, hidden=hidden, depth=depth).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    # Helper to compute position cost from q and goal position
    def pos_cost(q_np: np.ndarray, g_np: np.ndarray) -> float:
        p = fk.fk_pos(joint_names, q_np)
        e = p - g_np
        return float(np.dot(e, e))

    t0 = time.time()
    for it in range(1, iters + 1):
        # Sample batch of (q, t, goal)
        q_np = np.random.uniform(jmin, jmax, size=(batch, 7)).astype(np.float64)
        t_np = np.random.uniform(0.0, T, size=(batch, 1)).astype(np.float64)
        g_np = sample_goal_positions(batch)

        # Prepare tensors
        q = torch.tensor(q_np, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor(t_np / T, dtype=torch.float32, device=device)  # normalize to [0,1]
        g = torch.tensor(g_np, dtype=torch.float32, device=device)

        # Value
        x = build_input(q, t, g)
        V = model(x)  # (B,)

        # Gradients
        grad_q = torch.autograd.grad(V.sum(), q, create_graph=True, retain_graph=True)[0]  # (B,7)
        V_t = torch.autograd.grad(V.sum(), t, create_graph=True, retain_graph=True)[0].squeeze(-1)  # (B,)

        # Running cost requires FK; do FK in numpy loop (slower but OK for v1)
        # (Research-grade next step: vectorize via kinematics lib / batch FK)
        ell_np = np.zeros((batch,), dtype=np.float64)
        for i in range(batch):
            try:
                p = fk.fk_pos(joint_names, q_np[i])
            except Exception:
                # If FK fails (rare), set large cost
                ell_np[i] = 1e3
                continue
            e = p - g_np[i]
            ell_np[i] = Qp * float(np.dot(e, e))
        ell = torch.tensor(ell_np, dtype=torch.float32, device=device)

        # Hamiltonian min over u for dq=u:
        # min_u [ u^T R u + grad_q^T u ] = -1/4 * grad_q^T R^{-1} grad_q
        # => HJB residual: V_t + ell - 1/4 * grad_q^T R^{-1} grad_q = 0
        # R^{-1} diagonal:
        rinv = torch.tensor(R_inv[None, :], dtype=torch.float32, device=device)
        quad = (grad_q * rinv * grad_q).sum(dim=-1)  # (B,)
        pde_res = V_t + ell - 0.25 * quad
        loss_pde = (pde_res ** 2).mean()

        # Terminal condition V(q,T)=phi(q)=QpT*||p_ee(q)-g||^2
        # Sample a smaller terminal batch for stability
        bt = max(64, batch // 4)
        qT_np = np.random.uniform(jmin, jmax, size=(bt, 7)).astype(np.float64)
        gT_np = sample_goal_positions(bt)
        tT_np = np.ones((bt, 1), dtype=np.float64)  # normalized t=1

        # compute terminal phi via FK
        phi_np = np.zeros((bt,), dtype=np.float64)
        for i in range(bt):
            try:
                p = fk.fk_pos(joint_names, qT_np[i])
            except Exception:
                phi_np[i] = 1e3
                continue
            e = p - gT_np[i]
            phi_np[i] = QpT * float(np.dot(e, e))

        qT = torch.tensor(qT_np, dtype=torch.float32, device=device)
        tT = torch.tensor(tT_np, dtype=torch.float32, device=device)
        gT = torch.tensor(gT_np, dtype=torch.float32, device=device)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=device)

        VT = model(build_input(qT, tT, gT))
        loss_terminal = ((VT - phi) ** 2).mean()

        loss = loss_pde + 1.0 * loss_terminal

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if it % 50 == 0:
            dt_s = time.time() - t0
            rospy.loginfo(
                "iter=%d loss=%.4e (pde=%.4e term=%.4e) elapsed=%.1fs",
                it, float(loss.item()), float(loss_pde.item()), float(loss_terminal.item()), dt_s
            )

        if it % 500 == 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "in_dim": 11,
                    "hidden": hidden,
                    "depth": depth,
                    "T": T,
                    "dt": dt,
                    "R_diag": R_diag.tolist(),
                    "Qp": Qp,
                    "Qp_terminal": QpT,
                },
                out_path,
            )
            rospy.loginfo("Saved checkpoint: %s", out_path)

    # Final save
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_dim": 11,
            "hidden": hidden,
            "depth": depth,
            "T": T,
            "dt": dt,
            "R_diag": R_diag.tolist(),
            "Qp": Qp,
            "Qp_terminal": QpT,
        },
        out_path,
    )
    rospy.loginfo("Training complete. Saved: %s", out_path)


if __name__ == "__main__":
    main()
