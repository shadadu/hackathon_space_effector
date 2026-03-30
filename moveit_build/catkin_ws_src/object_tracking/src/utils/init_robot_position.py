#!/usr/bin/env python
import sys
import rospy
import moveit_commander
from geometry_msgs.msg import Pose


def initialize_panda_to_xyz(x, y, z,
                            group_name="panda_arm",
                            ee_link="panda_hand",
                            reference_frame="world",
                            planning_time=5.0,
                            num_attempts=5,
                            vel_scale=0.3,
                            acc_scale=0.3):
    """
    Move the Panda end effector to the provided XYZ position .

    Parameters
    ----------
    x, y, z : float
        Desired end-effector position.
    group_name : str
        MoveIt planning group 'panda_arm'.
    ee_link : str
        End effector link: 'panda_hand' .
    reference_frame : str
        Pose reference frame: 'world' or 'panda_link0'.
    """

    moveit_commander.roscpp_initialize(sys.argv)
    rospy.init_node("panda_initialize_xyz", anonymous=True)

    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()
    group = moveit_commander.MoveGroupCommander(group_name)

    rospy.sleep(1.0)  # allow interfaces to initialize

    group.set_pose_reference_frame(reference_frame)
    group.set_end_effector_link(ee_link)
    group.set_planning_time(planning_time)
    group.set_num_planning_attempts(num_attempts)
    group.set_max_velocity_scaling_factor(vel_scale)
    group.set_max_acceleration_scaling_factor(acc_scale)

    # Keep the current orientation, only change XYZ
    current_pose = group.get_current_pose(ee_link).pose

    pose_goal = Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    pose_goal.orientation = current_pose.orientation

    group.set_start_state_to_current_state()
    group.set_pose_target(pose_goal, ee_link)

    rospy.loginfo("Planning initialization move to x=%.3f y=%.3f z=%.3f", x, y, z)
    success = group.go(wait=True)

    group.stop()
    group.clear_pose_targets()

    if success:
        rospy.loginfo("Initialization move succeeded.")
    else:
        rospy.logwarn("Initialization move failed.")

    return success



def initialize_panda_to_xyz_with_group(group, x, y, z, ee_link="panda_hand"):
    current_pose = group.get_current_pose(ee_link).pose

    pose_goal = Pose()
    pose_goal.position.x = x
    pose_goal.position.y = y
    pose_goal.position.z = z
    pose_goal.orientation = current_pose.orientation

    group.set_start_state_to_current_state()
    group.set_pose_target(pose_goal, ee_link)

    success = group.go(wait=True)
    group.stop()
    group.clear_pose_targets()
    return success

if __name__ == "__main__":
    try:
        ok = initialize_panda_to_xyz(0.40, 0.00, 0.35)
        if not ok:
            rospy.logerr("Could not initialize Panda to requested XYZ pose.")
    except rospy.ROSInterruptException:
        pass
    finally:
        moveit_commander.roscpp_shutdown()