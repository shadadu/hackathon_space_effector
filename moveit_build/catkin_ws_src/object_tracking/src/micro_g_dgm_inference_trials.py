#!/usr/bin/env python3
"""Run reproducible, synthetic micro-g DGM rollouts and write one CSV row per trial."""

import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rospy
from geometry_msgs.msg import Point, Quaternion, Vector3
from moveit_commander import MoveGroupCommander
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import (
    GetPositionFK, GetPositionFKRequest, GetStateValidity,
    GetStateValidityRequest,
)
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

from object_tracking.dgm_jax import load_checkpoint
from object_tracking.fk_client import FKClient
from object_tracking.micro_g_dgm_rollout import (
    MicroGRolloutConfig, rollout_micro_g_dgm_persistent_policy,
)


TOLERANCES = [0.05, 0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
JMIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JMAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
DEFAULT_QDOT_MAX = np.array([1.5, 1.5, 1.5, 1.8, 1.8, 2.0, 2.0])


def vec(prefix, value):
    a = np.asarray(value, dtype=float).reshape(-1)
    return {f"{prefix}_{axis}": float(a[i]) for i, axis in enumerate("xyz"[:len(a)])}


def quat(prefix, value):
    return {f"{prefix}_{axis}": float(getattr(value, axis)) for axis in "xyzw"}


def numbered(prefix, value):
    return {f"{prefix}_{i + 1}": float(x) for i, x in enumerate(np.asarray(value).reshape(-1))}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def full_fk(fk, frame, ee_link, names, q):
    req = GetPositionFKRequest()
    req.header.frame_id = frame
    req.fk_link_names = [ee_link]
    req.robot_state.joint_state.name = list(names)
    req.robot_state.joint_state.position = [float(x) for x in q]
    response = fk(req)
    if response.error_code.val != 1 or not response.pose_stamped:
        raise RuntimeError(f"FK failed with code {response.error_code.val}")
    return response.pose_stamped[0].pose


def robot_state(names, q):
    state = RobotState()
    state.joint_state = JointState(name=list(names), position=[float(x) for x in q])
    return state


def state_valid(service, group_name, names, q):
    if service is None:
        return None
    req = GetStateValidityRequest(group_name=group_name)
    req.robot_state = robot_state(names, q)
    return bool(service(req).valid)


def sample_valid_q(rng, validity, group_name, names, attempts=1000):
    for _ in range(attempts):
        q = rng.uniform(JMIN, JMAX)
        if validity is None or state_valid(validity, group_name, names, q):
            return q
    raise RuntimeError(f"Could not sample a valid joint state in {attempts} attempts")


def sample_relative_position(rng, lo, hi, goal_tol, guard_width):
    # Mirrors sample_controlled_entry_pde's 2/3 global, 1/3 near-entry mixture.
    if rng.random() < 1.0 / 3.0:
        direction = rng.normal(size=3)
        direction /= max(np.linalg.norm(direction), 1e-12)
        r = direction * (goal_tol + guard_width) * np.cbrt(rng.random())
        return np.clip(r, lo, hi)
    return rng.uniform(lo, hi)


def make_odom(position, velocity, frame):
    msg = Odometry()
    msg.header.frame_id = frame
    msg.header.stamp = rospy.Time.now()
    msg.child_frame_id = "object_link"
    msg.pose.pose.position = Point(*[float(x) for x in position])
    msg.pose.pose.orientation = Quaternion(0.0, 0.0, 0.0, 1.0)
    msg.twist.twist.linear = Vector3(*[float(x) for x in velocity])
    return msg


def rms_norm(values):
    if not values:
        return math.nan
    return float(np.sqrt(np.mean([np.dot(x, x) for x in values])))


def first_index(mask):
    return next((i for i, value in enumerate(mask) if value), -1)


def main():
    rospy.init_node("micro_g_dgm_inference_trials")
    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")
    group = MoveGroupCommander(group_name)
    names = group.get_active_joints()
    if len(names) != 7:
        raise RuntimeError(f"Expected seven active joints, got {names}")

    package_dir = Path(__file__).resolve().parent.parent
    model_path = Path(rospy.get_param("~model_path", str(package_dir / "models/micro_g_dgm_v1.pkl")))
    results_dir = Path(rospy.get_param("~results_dir", str(package_dir / "models/results")))
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = Path(rospy.get_param("~results_path", str(results_dir / "micro_g_dgm_inference_trials.csv")))
    rospy.set_param("~out_path", str(results_dir / "micro_g_dgm_inference_step_path.csv"))

    M = int(rospy.get_param("~M", 100))
    if M <= 0:
        raise ValueError("~M must be positive")
    seed = int(rospy.get_param("~seed", 0))
    T = float(rospy.get_param("~T", 2.0))
    dt = float(rospy.get_param("~dt", 0.02))
    tolerances = [float(x) for x in rospy.get_param("~grasp_pos_tols", TOLERANCES)]
    if not tolerances or any(not np.isfinite(x) or x <= 0.0 for x in tolerances):
        raise ValueError("~grasp_pos_tols must contain positive finite values")
    grasp_vel_tol = float(rospy.get_param("~grasp_vel_tol", 0.5))
    base_min = np.array(rospy.get_param("~base_min", [-0.5, -0.5, -0.2]), dtype=float)
    base_max = np.array(rospy.get_param("~base_max", [0.5, 0.5, 0.5]), dtype=float)
    rel_min = np.array(rospy.get_param("~rel_min", [-0.6, -0.6, -0.6]), dtype=float)
    rel_max = np.array(rospy.get_param("~rel_max", [0.6, 0.6, 0.6]), dtype=float)
    vo_min = np.array(rospy.get_param("~vo_min", [-0.05, -0.05, -0.02]), dtype=float)
    vo_max = np.array(rospy.get_param("~vo_max", [0.05, 0.05, 0.02]), dtype=float)
    qdot_max = np.array(rospy.get_param("~joint_vel_limits", DEFAULT_QDOT_MAX.tolist()), dtype=float)
    bdot_max = np.array(rospy.get_param("~base_vel_limits", [0.08, 0.08, 0.08]), dtype=float)
    rq = np.array(rospy.get_param("~R_q_diag", [0.15] * 7), dtype=float)
    rb = np.array(rospy.get_param("~R_b_diag", [0.5] * 3), dtype=float)
    entry_guard = float(rospy.get_param("~entry_guard_width", 0.1))
    entry_weight = float(rospy.get_param("~entry_velocity_weight", 10.0))
    reach_min = float(rospy.get_param("~reach_min", 0.2))
    reach_max = float(rospy.get_param("~reach_max", 0.75))

    model, meta = load_checkpoint(str(model_path))
    checkpoint_hash = file_sha256(model_path)
    fk_name = rospy.get_param("~fk_service", "/compute_fk")
    rospy.wait_for_service(fk_name, timeout=float(rospy.get_param("~service_wait_timeout", 30.0)))
    fk_service = rospy.ServiceProxy(fk_name, GetPositionFK)
    fk_client = FKClient(fk_name, ee_link, world_frame)
    validity = None
    validity_name = rospy.get_param("~state_validity_service", "/check_state_validity")
    try:
        rospy.wait_for_service(validity_name, timeout=3.0)
        validity = rospy.ServiceProxy(validity_name, GetStateValidity)
    except rospy.ROSException:
        rospy.logwarn("State-validity service unavailable; validity fields will be blank")

    experiment_id = rospy.get_param("~experiment_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    rng = np.random.default_rng(seed)
    rows = []
    for start_id in range(M):
        q0 = sample_valid_q(rng, validity, group_name, names)
        b0 = rng.uniform(base_min, base_max)
        training_goal_tol = float(meta.get("goal_tol", 0.1))
        r0 = sample_relative_position(rng, rel_min, rel_max, training_goal_tol, entry_guard)
        ee0_pose = full_fk(fk_service, world_frame, ee_link, names, q0)
        ee0 = b0 + np.array([ee0_pose.position.x, ee0_pose.position.y, ee0_pose.position.z])
        p0 = ee0 - r0
        vo = rng.uniform(vo_min, vo_max)

        for tol in tolerances:
            cfg = MicroGRolloutConfig(
                T=T, dt=dt, joint_min=JMIN, joint_max=JMAX,
                joint_vel_limits=qdot_max, base_vel_limits=bdot_max,
                R_q_diag=rq, R_b_diag=rb, base_min=base_min, base_max=base_max,
                grasp_pos_tol=tol, grasp_vel_tol=grasp_vel_tol,
                entry_guard_width=entry_guard, entry_velocity_weight=entry_weight,
                reach_min=reach_min, reach_max=reach_max,
                require_final_reachable=False,
            )
            odom = make_odom(p0, vo, world_frame)
            started = time.perf_counter()
            error_type = error_message = ""
            termination = "unknown"
            step_records = []
            try:
                traj, q_hist, b_hist, _ = rollout_micro_g_dgm_persistent_policy(
                    model, q0, b0, odom, names, group, cfg, fk_client=fk_client,
                    step_records=step_records,
                )
                termination = "position_tolerance" if len(traj.joint_trajectory.points) < int(np.ceil(T / dt)) + 1 else "timeout"
            except Exception as exc:
                traj = None
                q_hist, b_hist = np.array([q0]), np.array([b0])
                termination, error_type, error_message = "exception", type(exc).__name__, str(exc)
            wall_s = time.perf_counter() - started

            qs = [q0]
            uq = []
            if traj is not None and traj.joint_trajectory.points:
                qs = [np.asarray(point.positions, dtype=float) for point in traj.joint_trajectory.points]
                uq = [np.asarray(point.velocities, dtype=float) for point in traj.joint_trajectory.points]
            count = min(len(qs), len(b_hist))
            qs, bs = qs[:count], [np.asarray(x) for x in b_hist[:count]]
            if not qs:
                qs, bs = [q0], [b0]
            # The legacy history arrays contain pre-integration states. Include the
            # final integrated state so closest approach really means the closest
            # state reached, including the step which crossed the stopping radius.
            if step_records:
                final_step = step_records[-1]
                if not np.allclose(qs[-1], final_step["q_after"]) or not np.allclose(bs[-1], final_step["b_after"]):
                    qs.append(final_step["q_after"])
                    bs.append(final_step["b_after"])
            times = [min(i * dt, T) for i in range(len(qs))]
            object_positions = [p0 + vo * t for t in times]
            ee_poses = [full_fk(fk_service, world_frame, ee_link, names, q) for q in qs]
            ee_positions = [b + np.array([p.position.x, p.position.y, p.position.z]) for b, p in zip(bs, ee_poses)]
            rs = [ee - obj for ee, obj in zip(ee_positions, object_positions)]
            rnorms = np.array([np.linalg.norm(x) for x in rs])
            imin = int(np.argmin(rnorms))
            base_distances = np.array([np.linalg.norm(b - obj) for b, obj in zip(bs, object_positions)])
            ibase = int(np.argmin(base_distances))
            ub = [record["u_b"] for record in step_records]
            uq_used = uq[:len(qs)]
            vrel = []
            for i, q in enumerate(qs[:len(uq_used)]):
                jac = np.asarray(group.get_jacobian_matrix(q.tolist()))[:3, :7]
                base_u = ub[i] if i < len(ub) else np.zeros(3)
                vrel.append(jac.dot(uq_used[i]) + base_u - vo)
            position_ready = [x <= tol for x in rnorms]
            velocity_ready = [np.linalg.norm(x) <= grasp_vel_tol for x in vrel]
            grasp_ready = [position_ready[i] and i < len(velocity_ready) and velocity_ready[i] for i in range(len(position_ready))]
            first_pos, first_grasp = first_index(position_ready), first_index(grasp_ready)
            end_valid = state_valid(validity, group_name, names, qs[-1])
            joint_margin = float(np.min(np.minimum(np.asarray(qs) - JMIN, JMAX - np.asarray(qs))))
            row = {
                "experiment_id": experiment_id, "trial_id": len(rows), "start_id": start_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "model_path": str(model_path), "model_sha256": checkpoint_hash,
                "model_architecture": meta.get("architecture", ""), "model_metadata_json": json.dumps(meta, sort_keys=True, default=json_default),
                "T": T, "dt": dt, "N": int(np.ceil(T / dt)) + 1, "TOL": tol,
                "grasp_pos_tol": tol, "grasp_vel_tol": grasp_vel_tol,
                "success": bool(any(grasp_ready)), "termination_reason": termination,
                "exception_type": error_type, "exception_message": error_message,
                "elapsed_rollout_s": wall_s, "simulated_time_last": times[-1],
                "steps_executed": len(step_records), "k_last": step_records[-1]["k"] if step_records else 0,
                "last_valid_k": len(qs) - 1,
                "norm_r_start": rnorms[0], "norm_r_min": rnorms[imin], "norm_r_last": rnorms[-1],
                "distance_reduction": rnorms[0] - rnorms[imin], "k_r_min": imin, "time_to_min_r": times[imin],
                "relative_speed_at_min_r": np.linalg.norm(vrel[imin]) if imin < len(vrel) else math.nan,
                "relative_speed_last": np.linalg.norm(vrel[-1]) if vrel else math.nan,
                "position_ready": bool(any(position_ready)), "velocity_ready": bool(any(velocity_ready)),
                "velocity_ready_last": bool(velocity_ready[-1]) if velocity_ready else False,
                "first_position_ready_k": first_pos, "first_grasp_ready_k": first_grasp,
                "time_to_position_ready": times[first_pos] if first_pos >= 0 else math.nan,
                "time_to_grasp_ready": times[first_grasp] if first_grasp >= 0 else math.nan,
                "object_base_distance_start": base_distances[0], "object_base_distance_min": base_distances[ibase],
                "object_base_distance_last": base_distances[-1], "k_base_distance_min": ibase,
                "final_reach_shell_valid": bool(reach_min <= base_distances[-1] <= reach_max),
                "start_state_valid": state_valid(validity, group_name, names, q0), "end_state_valid": end_valid,
                "minimum_joint_limit_margin": joint_margin,
                "joint_saturation_count": sum(np.any(np.abs(x) >= qdot_max - 1e-9) for x in uq_used),
                "base_saturation_count": sum(np.any(np.abs(x) >= bdot_max - 1e-9) for x in ub),
                "joint_saturation_fraction": sum(np.any(np.abs(x) >= qdot_max - 1e-9) for x in uq_used) / max(1, len(uq_used)),
                "base_saturation_fraction": sum(np.any(np.abs(x) >= bdot_max - 1e-9) for x in ub) / max(1, len(ub)),
                "max_norm_u_q": max([np.linalg.norm(x) for x in uq_used], default=math.nan),
                "rms_norm_u_q": rms_norm(uq_used), "max_norm_u_b": max([np.linalg.norm(x) for x in ub], default=math.nan),
                "rms_norm_u_b": rms_norm(ub), "nan_hits": sum(record["nonfinite_control"] for record in step_records),
                "initial_in_distribution": bool(np.all(q0 >= JMIN) and np.all(q0 <= JMAX) and np.all(b0 >= base_min) and np.all(b0 <= base_max) and np.all(r0 >= rel_min) and np.all(r0 <= rel_max) and np.all(vo >= vo_min) and np.all(vo <= vo_max)),
                "odometry_age_start_s": 0.0,
            }
            row.update(vec("start_position", p0)); row.update(quat("start_orientation", odom.pose.pose.orientation)); row.update(vec("start_velocity", vo))
            row.update(vec("end_position", object_positions[-1])); row.update(quat("end_orientation", odom.pose.pose.orientation)); row.update(vec("end_velocity", vo))
            row.update(vec("r_start", rs[0])); row.update(vec("r_min", rs[imin])); row.update(vec("r_last", rs[-1]))
            row.update(vec("base_start", bs[0])); row.update(vec("base_end", bs[-1])); row.update(vec("base_closest_to_object", bs[ibase]))
            row.update(numbered("q_start", q0)); row.update(numbered("q_end", qs[-1]))
            row.update(quat("ee_orientation_start", ee_poses[0].orientation)); row.update(quat("ee_orientation_end", ee_poses[-1].orientation))
            rows.append(row)
            rospy.loginfo("Trial %d tol %.3f: min |r|=%.4f success=%s", start_id, tol, rnorms[imin], row["success"])

    write_header = not result_path.exists() or result_path.stat().st_size == 0
    with result_path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    rospy.loginfo("Wrote %d trial rows to %s", len(rows), result_path)


if __name__ == "__main__":
    main()
