#!/usr/bin/env python3
import rospy
import numpy as np

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_commander import RobotCommander

from geometry_msgs.msg import Point

# Jacobian service
try:
    from jacobian_server.srv import GetJacobian, GetJacobianRequest
    HAS_JACOBIAN_SRV = True
except Exception:
    HAS_JACOBIAN_SRV = False


def interpolate_joints(q0, q1, n=50, duration=2.0):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    pts = []
    for i in range(n):
        a = i / max(n - 1, 1)
        q = (1 - a) * q0 + a * q1
        t = a * duration
        pts.append((q.tolist(), t))
    return pts


class DGMPlannerService:
    def __init__(self):
        self.robot = RobotCommander()

        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")
        self.world_frame = rospy.get_param("~world_frame", "world")

        # IK service
        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")
        rospy.wait_for_service(self.ik_service, timeout=50.0)
        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)

        # Jacobian service (optional but recommended)
        self.jacobian_service = rospy.get_param("~jacobian_service", "/get_jacobian")
        self.get_jacobian = None
        if HAS_JACOBIAN_SRV:
            try:
                rospy.loginfo("Waiting for Jacobian service: %s", self.jacobian_service)
                rospy.wait_for_service(self.jacobian_service, timeout=20.0)
                self.get_jacobian = rospy.ServiceProxy(self.jacobian_service, GetJacobian)
                rospy.loginfo("Jacobian service connected: %s", self.jacobian_service)
            except Exception as e:
                rospy.logwarn("Jacobian service not available (%s). Continuing without it.", str(e))
        else:
            rospy.logwarn("jacobian_server.srv not importable in this environment. "
                          "Did you build/source jacobian_server? Continuing without it.")

        # DGM service endpoint
        self.service_name = rospy.get_param("~service_name", "/dgm/get_motion_plan")
        self.srv = rospy.Service(self.service_name, GetMotionPlan, self.handle)

        rospy.loginfo("DGM planner service up: %s (IK: %s)", self.service_name, self.ik_service)

    def query_jacobian(self, group_name, ee_link, joint_names, joint_positions, reference_point=(0.0, 0.0, 0.0)):
        """
        Returns J as numpy array (rows x cols), usually 6 x N.
        """
        if self.get_jacobian is None:
            return None

        req = GetJacobianRequest()
        req.group_name = group_name
        req.link_name = ee_link
        req.joint_names = list(joint_names)
        req.joint_positions = list(joint_positions)
        req.reference_point = Point(*reference_point)

        resp = self.get_jacobian(req)
        if resp.message != "OK" or resp.rows <= 0 or resp.cols <= 0:
            raise rospy.ServiceException(f"Jacobian service failed: {resp.message}")

        J = np.array(resp.jacobian, dtype=float).reshape(resp.rows, resp.cols)
        return J

    def handle(self, req):
        """
        req.motion_plan_request is a MotionPlanRequest
        Return GetMotionPlanResponse containing MotionPlanResponse
        """
        mpr = req.motion_plan_request

        resp = MotionPlanResponse()
        resp.error_code = MoveItErrorCodes()
        resp.planning_time = 0.0

        if not mpr.goal_constraints:
            resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Extract goal pose from first PositionConstraint primitive_pose
        try:
            pc = mpr.goal_constraints[0].position_constraints[0]
            goal_pose = pc.constraint_region.primitive_poses[0]
            goal_frame = pc.header.frame_id if pc.header.frame_id else self.world_frame
        except Exception:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        # IK request
        ikreq = GetPositionIKRequest()
        ikreq.ik_request.group_name = mpr.group_name or self.group_name
        ikreq.ik_request.ik_link_name = self.ee_link
        ikreq.ik_request.pose_stamped.header.frame_id = goal_frame
        ikreq.ik_request.pose_stamped.pose = goal_pose

        # Seed state
        if mpr.start_state and mpr.start_state.joint_state.name:
            start_state = mpr.start_state
        else:
            start_state = self.robot.get_current_state()

        ikreq.ik_request.robot_state = start_state
        ikreq.ik_request.timeout = rospy.Duration(0.2)

        t0 = rospy.Time.now()
        ikresp = self.ik(ikreq)
        resp.planning_time = (rospy.Time.now() - t0).to_sec()

        if ikresp.error_code.val != MoveItErrorCodes.SUCCESS:
            resp.error_code.val = ikresp.error_code.val
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Active joints in group order
        group = self.robot.get_group(mpr.group_name or self.group_name)
        active_joints = group.get_active_joints()

        # Joint maps
        start_map = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
        goal_map = dict(zip(ikresp.solution.joint_state.name, ikresp.solution.joint_state.position))

        try:
            q0 = [start_map[j] for j in active_joints]
            q1 = [goal_map[j] for j in active_joints]
        except KeyError:
            resp.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
            return GetMotionPlanResponse(motion_plan_response=resp)

        # ---- NEW: Jacobian calls (start + goal) ----
        # This is where you'll later use HJB/DGM policy:
        # u* = -R^{-1}(V_q + J^T V_r)
        try:
            J0 = self.query_jacobian(mpr.group_name or self.group_name, self.ee_link, active_joints, q0)
            J1 = self.query_jacobian(mpr.group_name or self.group_name, self.ee_link, active_joints, q1)
            if J0 is not None:
                Jlin0 = J0[0:3, :]
                rospy.loginfo_throttle(2.0, "Jacobian(start) shape=%s, lin-norm=%.3f",
                                       str(J0.shape), float(np.linalg.norm(Jlin0)))
            if J1 is not None:
                Jlin1 = J1[0:3, :]
                rospy.loginfo_throttle(2.0, "Jacobian(goal) shape=%s, lin-norm=%.3f",
                                       str(J1.shape), float(np.linalg.norm(Jlin1)))
        except Exception as e:
            rospy.logwarn_throttle(2.0, "Jacobian query failed: %s", str(e))

        # Placeholder planner: interpolation (keep for now)
        npts = int(rospy.get_param("~n_points", 60))
        duration = float(rospy.get_param("~duration", 2.0))
        pts = interpolate_joints(q0, q1, n=npts, duration=duration)

        traj = RobotTrajectory()
        traj.joint_trajectory.joint_names = active_joints
        for q, t in pts:
            p = JointTrajectoryPoint()
            p.positions = q
            p.time_from_start = rospy.Duration.from_sec(t)
            traj.joint_trajectory.points.append(p)

        resp.trajectory = traj
        resp.error_code.val = MoveItErrorCodes.SUCCESS
        return GetMotionPlanResponse(motion_plan_response=resp)


if __name__ == "__main__":
    rospy.init_node("dgm_planner_node")
    DGMPlannerService()
    rospy.spin()