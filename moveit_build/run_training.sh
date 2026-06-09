# inside moveit container
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
export ROS_MASTER_URI=http://ros_master:11311
unset ROS_HOSTNAME
export ROS_IP=$(hostname -i | awk '{print $1}')

rosrun object_tracking dgm_training.py _T:=2.0 _dt:=0.02 _iters:=4000 _batch:=256
