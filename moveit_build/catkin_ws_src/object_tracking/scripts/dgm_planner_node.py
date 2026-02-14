#!/usr/bin/env python3
import rospy
import numpy as np
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanResponse
from moveit_msgs.msg import MotionPlanResponse, MoveItErrorCodes, RobotTrajectory
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from trajectory_msgs.msg import JointTrajectoryPoint
from moveit_commander import RobotCommander


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

        # IK service name varies; discover via: rosservice list | grep compute_ik
        self.ik_service = rospy.get_param("~ik_service", "/compute_ik")

        rospy.wait_for_service(self.ik_service, timeout=50.0)
        self.ik = rospy.ServiceProxy(self.ik_service, GetPositionIK)

        # DGM service endpoint
        self.service_name = rospy.get_param("~service_name", "/dgm/get_motion_plan")
        self.srv = rospy.Service(self.service_name, GetMotionPlan, self.handle)

        rospy.loginfo("DGM planner service up: %s (IK: %s)", self.service_name, self.ik_service)

    def handle(self, req):
        """
        req.motion_plan_request is a MotionPlanRequest
        Return GetMotionPlanResponse containing MotionPlanResponse
        """
        mpr = req.motion_plan_request

        resp = MotionPlanResponse()
        resp.error_code = MoveItErrorCodes()
        resp.planning_time = 0.0

        # Must have a goal constraint (pose/orientation constraints etc.)
        if not mpr.goal_constraints:
            resp.error_code.val = MoveItErrorCodes.INVALID_MOTION_PLAN
            return GetMotionPlanResponse(motion_plan_response=resp)

        # For the placeholder, we support goal pose constraints created in benchmark_runner:
        # We'll extract the pose from the first PositionConstraint primitive_pose.
        try:
            pc = mpr.goal_constraints[0].position_constraints[0]
            goal_pose = pc.constraint_region.primitive_poses[0]
            goal_frame = pc.header.frame_id if pc.header.frame_id else self.world_frame
        except Exception as e:
            resp.error_code.val = MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Build IK request
        ikreq = GetPositionIKRequest()
        ikreq.ik_request.group_name = mpr.group_name or self.group_name
        ikreq.ik_request.ik_link_name = self.ee_link
        ikreq.ik_request.pose_stamped.header.frame_id = goal_frame
        ikreq.ik_request.pose_stamped.pose = goal_pose

        # Seed state: start_state if provided, else current
        if mpr.start_state and mpr.start_state.joint_state.name:
            ikreq.ik_request.robot_state = mpr.start_state
        else:
            ikreq.ik_request.robot_state = self.robot.get_current_state()

        ikreq.ik_request.timeout = rospy.Duration(0.2)
        # Noetic moveit_msgs/PositionIKRequest has no 'attempts' field.

        t0 = rospy.Time.now()
        ikresp = self.ik(ikreq)
        resp.planning_time = (rospy.Time.now() - t0).to_sec()

        if ikresp.error_code.val != MoveItErrorCodes.SUCCESS:
            resp.error_code.val = ikresp.error_code.val
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Extract start joints for the group (simple approach: use group active joints order)
        active_joints = self.robot.get_group(mpr.group_name or self.group_name).get_active_joints()

        start_state = mpr.start_state if (mpr.start_state and mpr.start_state.joint_state.name) else self.robot.get_current_state()

        # Map joint names to positions
        start_map = dict(zip(start_state.joint_state.name, start_state.joint_state.position))
        goal_map = dict(zip(ikresp.solution.joint_state.name, ikresp.solution.joint_state.position))

        # Ensure we have all joints
        try:
            q0 = [start_map[j] for j in active_joints]
            q1 = [goal_map[j] for j in active_joints]
        except KeyError:
            resp.error_code.val = MoveItErrorCodes.INVALID_ROBOT_STATE
            return GetMotionPlanResponse(motion_plan_response=resp)

        # Placeholder "planner": joint interpolation (replace with DGM trajectory)
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
