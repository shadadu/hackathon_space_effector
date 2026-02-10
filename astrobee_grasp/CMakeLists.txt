cmake_minimum_required(VERSION 3.0.2)
project(astrobee_grasp)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  nav_msgs
  tf2_ros
  std_msgs
)

catkin_package()

catkin_install_python(
  PROGRAMS
    nodes/pose_tracker.py
    nodes/ekf_node.py
    nodes/tf_broadcaster.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
