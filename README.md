# Build (run in the hackathon_space_effector directory, where the Dockerfile is located)
1. docker build -t astrobee_grasp:noetic .
2. 
# How to launch with Docker
1. docker run --rm --net=host astrobee_grasp:noetic
2. source /opt/ros/noetic/setup.bash
3. source ~/catkin_ws/devel/setup.bash
4. roslaunch astrobee_grasp perception.launch

# To exit
1. Ctrl + C (To exit ros launch)
2. exit (to exit docker)

    


