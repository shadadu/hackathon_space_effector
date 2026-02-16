cmake_minimum_required(VERSION 3.0.2)
project(jacobian_server)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  message_generation
)

find_package(moveit_core REQUIRED)
find_package(Eigen3 REQUIRED)

add_service_files(
  FILES
  GetJacobian.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp geometry_msgs message_runtime
  DEPENDS moveit_core EIGEN3
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${moveit_core_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(jacobian_server_node src/jacobian_server_node.cpp)
add_dependencies(jacobian_server_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(jacobian_server_node ${catkin_LIBRARIES} ${moveit_core_LIBRARIES})