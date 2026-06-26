cmake_minimum_required(VERSION 3.0.2)
project(object_tracking)

find_package(catkin REQUIRED COMPONENTS
  rospy
  nav_msgs
  std_msgs
  sensor_msgs
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  shape_msgs
  std_msgs
  tf2_ros
  tf2_geometry_msgs
  message_generation
)

catkin_python_setup()
## Messages
add_message_files(
  FILES
    InterceptMetrics.msg
    BenchmarkSummary.msg
)

## Services
add_service_files(
  FILES
    StartTrial.srv
    RunBenchmark.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
   message_runtime
    rospy
    std_msgs
    geometry_msgs
    moveit_ros_planning_interface
    nav_msgs
    sensor_msgs
    moveit_msgs
    shape_msgs
    tf2_ros
    tf2_geometry_msgs
    message_runtime
)


catkin_install_python(
  PROGRAMS
    src/object_to_planning_scene.py
    src/dgm_planner_node.py
    src/benchmark_runner.py
    src/benchmark_100_trials.py
    src/intercept_planner.py
    src/intercept_evaluator.py
    src/trajectory_executor_manager.py
    src/dgm_planner_node.py
    src/dgm_train.py
    src/dgm_training.py
    src/dgm_pretrain.py
    src/dgm_dataset_gen.py
    src/micro_g_dgm_intercept_planner.py
    src/micro_g_dgm_planner_node.py
    src/micro_g_dgm_training.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)