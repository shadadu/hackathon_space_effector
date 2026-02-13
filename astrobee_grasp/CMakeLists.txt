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
    scripts/pose_tracker.py
    scripts/object_ekf.py
    scripts/tf_broadcaster.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
