# Build (run in the hackathon_space_effector directory, where the Dockerfile is located)
1. docker build -t astrobee_grasp:noetic .
2. 
# How to launch with Docker
1. docker run --rm --net=host astrobee_grasp:noetic bash
2. source /opt/ros/noetic/setup.bash
3. source ~/catkin_ws/devel/setup.bash
4. roslaunch astrobee_grasp perception.launch

Some start errors could be lack of memory issues
Check docker disk usage: 
1. docker system df
2. docker system prune -a --volumes (or restart docker)

# To exit ROS and Docker 
(since CMD ["roslaunch", "astrobee_grasp", "perception.launch"] is added to Dockerfile)
1. Ctrl + C

    


