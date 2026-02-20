# Space Effector
An implementation of a robotic end-effector for micro-gravity environments to intercept and grasp objects. 
Gearing for grasp and move activities inside or outside space stations and modules.  

Stack: 
1. NASA Astrobee: Github https://github.com/nasa/astrobee, Main site: https://www.nasa.gov/astrobee/
2. MoveIt: https://moveit.picknik.ai/main/index.html

# Running the services
The stack provides a simulation platform for researching and testing micro-gravity environment Optimal Control and Deep Learning based trajectory planners and executors.
To start:
1. In the project root dir, run ./start.sh to boot up the docker containers: 
    rosnet(bridge network), ros_master, astrobee_grasp, moveit
    start.sh brings up the stack and runs tests on the environment, and test that
    the servers can communicate via ROS and send and receive messages. 
    It also runs diagnostic tests of the pose, intercept, and trajectory planners. 
2. Run ./stop.sh to stop the services and tear down the stack

# Training DGM
1. rosrun object_tracking dgm_pretrain.py _T:=2.0 _iters:=3000 _batch:=192

    


