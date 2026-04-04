#!/usr/bin/env python3
import os
import time
import numpy as np
import rospy
import torch
import torch.optim as optim

from moveit_commander import MoveGroupCommander
from object_tracking.dgm_model import DGMValueNet, build_input, save_checkpoint
from object_tracking.hjb_loss import hjb_residual_loss, terminal_loss
from object_tracking.fk_client import FKClient


def panda_joint_limits():
    jmin = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
    jmax = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
    return jmin, jmax


def sample_goals(n):
    xs = np.random.uniform(0.25, 0.65, (n, 1))
    ys = np.random.uniform(-0.30, 0.30, (n, 1))
    zs = np.random.uniform(0.10, 0.60, (n, 1))
    return np.hstack([xs, ys, zs]).astype(np.float64)


def position_loss_fn(fk, joint_names, batch, Qp, g_np, q_np):
    l_np = np.zeros((batch,), dtype=np.float64)
    for i in range(batch):
        try:
            p = fk.ee_position(joint_names, q_np[i])
            e = p - g_np[i]
            l_np[i] = Qp * float(np.dot(e, e))
        except Exception:
            rospy.logwarn("fk_pos: couldn't retrieve fk position")
            l_np[i] = 1e3
    rospy.loginfo("Loss vector %s", l_np)
    return l_np


def main():
    rospy.init_node("dgm_pretrain")

    print(f'torch version: {torch.__version__}')
    print(f'numpy version: {np.__version__}')
    device = rospy.get_param("~device", "cpu")
    T = float(rospy.get_param("~T", 2.0))

    iters = int(rospy.get_param("~iters", 3000))
    batch = int(rospy.get_param("~batch", 192))
    lr = float(rospy.get_param("~lr", 3e-4))
    hidden = int(rospy.get_param("~hidden", 256))
    depth = int(rospy.get_param("~depth", 4))
    print(f'device: {device}')

    Qp = float(rospy.get_param("~Qp", 10.0))
    QpT = float(rospy.get_param("~Qp_terminal", 80.0))
    R_diag = torch.Tensor(rospy.get_param("~R_diag", [0.15] * 7))
    print(f'R_diag {R_diag}')
    # R_diag = np.asarray(R_diag_inp, dtype=np.float64)
    R_inv_diag = (1.0 / torch.max(R_diag))

    out_path = rospy.get_param("~out_path", "/root/catkin_ws/src/object_tracking/models/panda_dgm_v1.pth")

    group = MoveGroupCommander("panda_arm")
    joint_names = group.get_active_joints()
    if len(joint_names) != 7:
        raise RuntimeError(f"Expected 7 joints, got {len(joint_names)}: {joint_names}")

    # client to retrieve position of end effector to compute running loss function term
    fk = FKClient(service="/compute_fk", ee_link="panda_hand", frame="world")

    jmin, jmax = panda_joint_limits()

    model = DGMValueNet(in_dim=11, hidden=hidden, depth=depth).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    t0 = time.time()
    loss = 0.0
    for it in range(1, iters + 1):
        rospy.loginfo("Running iter %s: ", it)
        q_np = np.random.uniform(jmin, jmax, (batch, 7)).astype(np.float64)
        t_np = np.random.uniform(0.0, T, (batch, 1)).astype(np.float64)
        g_np = sample_goals(batch)

        # running cost via FK (position-only)
        l_np = position_loss_fn(fk, joint_names, batch, Qp, g_np, q_np)
        # l_np = np.zeros((batch,), dtype=np.float64)
        # for i in range(batch):
        #     try:
        #         p = fk.ee_position(joint_names, q_np[i])
        #         e = p - g_np[i]
        #         l_np[i] = Qp * float(np.dot(e, e))
        #     except Exception:
        #         rospy.logwarn("l_np: couldn't retrieve fk position")
        #         l_np[i] = 1e3

        q = torch.tensor(q_np, dtype=torch.float32, device=device, requires_grad=True)
        t = torch.tensor((t_np / T), dtype=torch.float32, device=device, requires_grad=True)
        g = torch.tensor(g_np, dtype=torch.float32, device=device)
        l = torch.tensor(l_np, dtype=torch.float32, device=device)

        V = model(build_input(q, t, g))
        loss_pde = hjb_residual_loss(V, q, t, l,
                                     R_inv_diag)  # hjb_residual_loss(V, q, t_norm, running_cost, R_inv_diag)

        # terminal batch
        bt = max(64, batch // 3)
        qT_np = np.random.uniform(jmin, jmax, (bt, 7)).astype(np.float64)
        gT_np = sample_goals(bt)

        phi_np = position_loss_fn(fk, joint_names, bt, QpT, gT_np, qT_np)

        # phi_np = np.zeros((bt,), dtype=np.float64)
        # for i in range(bt):
        #     try:
        #         p = fk.ee_position(joint_names, qT_np[i])
        #         e = p - gT_np[i]
        #         phi_np[i] = QpT * float(np.dot(e, e))
        #     except Exception:
        #         rospy.logwarn("phi_np: couldn't retrieve fk position")
        #         phi_np[i] = 1e3

        qT = torch.tensor(qT_np, dtype=torch.float32, device=device)
        tT = torch.ones((bt, 1), dtype=torch.float32, device=device, requires_grad=True)  # normalized t=1
        gT = torch.tensor(gT_np, dtype=torch.float32, device=device)
        phi = torch.tensor(phi_np, dtype=torch.float32, device=device)

        VT = model(build_input(qT, tT, gT))
        loss_term = terminal_loss(VT, phi)

        loss = loss_pde + loss_term

        print(f'loss values: loss_pde {loss_pde}, loss_term {loss_term}, loss {loss}')

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if it % 50 == 0:
            rospy.loginfo("iter=%d loss=%.3e pde=%.3e term=%.3e elapsed=%.1fs",
                          it, float(loss.item()), float(loss_pde.item()), float(loss_term.item()), time.time() - t0)

        if it % 500 == 0:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            checkpoint = {
                'epoch': it + 1,  # Save the next epoch number to start from
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss,

            }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    checkpoint = {
        'epoch': 0 + 1,  # Save the next epoch number to start from
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'loss': loss,

    }
    torch.save(checkpoint, out_path)
    # save_checkpoint(out_path, model, {
    #     "in_dim": 11, "hidden": hidden, "depth": depth,
    #     "T": T, "R_diag": R_diag.tolist(),
    #     "Qp": Qp, "Qp_terminal": QpT,
    # })
    # # rospy.loginfo("Saved: %s", out_path)
    rospy.loginfo("DONE. Saved final: %s", out_path)


if __name__ == "__main__":
    main()
