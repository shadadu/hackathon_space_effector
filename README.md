# Build (run in the hackathon_space_effector directory, where the Dockerfile is located)
1. docker build -t astrobee_grasp:noetic .

# How to launch with Docker
1. docker run -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -e QT_X11_NO_MITSHM=1 \
  -p 11311:11311 \
  astrobee_grasp:noetic \
  bash

2. source /opt/ros/noetic/setup.bash
   source ~/catkin_ws/devel/setup.bash
3. roslaunch astrobee_grasp perception.launch

Some start errors could be lack of memory issues
Check docker disk usage: 
1. docker system df
2. docker system prune -a --volumes (or restart docker)

# To exit ROS and Docker 
(since CMD ["roslaunch", "astrobee_grasp", "perception.launch"] is added to Dockerfile)
1. Ctrl + C

    


