cmake_minimum_required(VERSION 3.0.2)
project(object_tracking)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
)

catkin_python_setup()

catkin_package()

catkin_install_python(
  PROGRAMS
    scripts/object_to_planning_scene.py
    scripts/dgm_planner_node.py
    scripts/benchmark_runner.py
    scripts/intercept_planner.py
    scripts/dgm_pretrain.py
    scripts/dgm_dataset_gen.py
    scripts/benchmark_100_trials.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)