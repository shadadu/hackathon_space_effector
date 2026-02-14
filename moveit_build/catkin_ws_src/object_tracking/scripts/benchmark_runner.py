#!/usr/bin/env python3
import rospy
from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from moveit_msgs.msg import RobotState
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from moveit_commander import RobotCommander
from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest
from moveit_msgs.msg import PlanningSceneComponents
import numpy as np


def make_goal_constraints_from_pose(goal_pose: PoseStamped,
                                   link_name: str,
                                   pos_tol: float = 0.01,
                                   ang_tol: float = 0.05) -> Constraints:
    """
    Create a MoveIt Constraints message (position + orientation constraints)
    from a PoseStamped goal.
    """
    c = Constraints()
    c.name = "goal_pose_constraints"

    # PositionConstraint uses a bounding volume; simplest is a small BOX around goal position
    pc = PositionConstraint()
    pc.header = goal_pose.header
    pc.link_name = link_name
    pc.target_point_offset.x = 0.0
    pc.target_point_offset.y = 0.0
    pc.target_point_offset.z = 0.0
    pc.weight = 1.0

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [pos_tol * 2, pos_tol * 2, pos_tol * 2]

    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal_pose.pose)

    # OrientationConstraint
    oc = OrientationConstraint()
    oc.header = goal_pose.header
    oc.link_name = link_name
    oc.orientation = goal_pose.pose.orientation
    oc.absolute_x_axis_tolerance = ang_tol
    oc.absolute_y_axis_tolerance = ang_tol
    oc.absolute_z_axis_tolerance = ang_tol
    oc.weight = 1.0

    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)
    return c


def get_planning_scene(service_name: str = "/get_planning_scene"):
    """
    Optional: fetch planning scene snapshot. Useful for ensuring both planners
    see the same world when you evolve the benchmark harness.
    """
    rospy.wait_for_service(service_name, timeout=10.0)
    srv = rospy.ServiceProxy(service_name, GetPlanningScene)

    req = GetPlanningSceneRequest()
    req.components.components = (
        PlanningSceneComponents.SCENE_SETTINGS |
        PlanningSceneComponents.ROBOT_STATE |
        PlanningSceneComponents.WORLD_OBJECT_NAMES |
        PlanningSceneComponents.WORLD_OBJECT_GEOMETRY |
        PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
    )
    return srv(req).scene


def call_moveit_get_motion_plan(service_name: str, mpr: MotionPlanRequest):
    rospy.wait_for_service(service_name, timeout=10.0)
    proxy = rospy.ServiceProxy(service_name, GetMotionPlan)
    req = GetMotionPlanRequest()
    req.motion_plan_request = mpr
    resp = proxy(req)
    return resp.motion_plan_response


def main():
    rospy.init_node("benchmark_runner")

    # ---- Params you must set to match your robot ----
    group_name = rospy.get_param("~group_name", "panda_arm")
    ee_link = rospy.get_param("~ee_link", "panda_hand")
    world_frame = rospy.get_param("~world_frame", "world")

    # MoveIt services (discover with: rosservice list | grep plan_kinematic_path)
    ompl_service = rospy.get_param("~ompl_service", "/plan_kinematic_path")
    dgm_service  = rospy.get_param("~dgm_service",  "/dgm/get_motion_plan")

    # A simple goal pose for initial testing (you can replace with grasp pose logic)
    goal = PoseStamped()
    goal.header.frame_id = world_frame
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = rospy.get_param("~goal_x", 0.5)
    goal.pose.position.y = rospy.get_param("~goal_y", 0.0)
    goal.pose.position.z = rospy.get_param("~goal_z", 0.4)
    goal.pose.orientation.w = 1.0

    # ---- Build MotionPlanRequest ----
    robot = RobotCommander()
    start_state = robot.get_current_state()  # moveit_msgs/RobotState

    mpr = MotionPlanRequest()
    mpr.group_name = group_name
    mpr.start_state = start_state
    mpr.goal_constraints = [make_goal_constraints_from_pose(goal, ee_link)]
    mpr.allowed_planning_time = rospy.get_param("~allowed_planning_time", 2.0)
    mpr.num_planning_attempts = int(rospy.get_param("~num_planning_attempts", 5))
    mpr.max_velocity_scaling_factor = rospy.get_param("~vel_scale", 0.3)
    mpr.max_acceleration_scaling_factor = rospy.get_param("~acc_scale", 0.3)

    # ---- Call baseline OMPL ----
    rospy.loginfo("Calling OMPL service: %s", ompl_service)
    ompl_resp = call_moveit_get_motion_plan(ompl_service, mpr)
    rospy.loginfo("OMPL error_code=%d planning_time=%.3f",
                  ompl_resp.error_code.val, ompl_resp.planning_time)

    # ---- Call DGM service ----
    rospy.loginfo("Calling DGM service: %s", dgm_service)
    dgm_resp = call_moveit_get_motion_plan(dgm_service, mpr)
    rospy.loginfo("DGM error_code=%d planning_time=%.3f",
                  dgm_resp.error_code.val, dgm_resp.planning_time)

    # ---- Basic trajectory sanity ----
    if ompl_resp.error_code.val == 1:
        rospy.loginfo("OMPL points: %d",
                      len(ompl_resp.trajectory.joint_trajectory.points))
    if dgm_resp.error_code.val == 1:
        rospy.loginfo("DGM points: %d",
                      len(dgm_resp.trajectory.joint_trajectory.points))


if __name__ == "__main__":
    main()
