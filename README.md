# Space Effector
An implementation of a robotic end-effector for micro-gravity environments to intercept and grasp objects. 
Gearing for grasp and move activities inside or outside space stations and modules.  

Stack: 
1. NASA Astrobee: Github https://github.com/nasa/astrobee, Main site: https://www.nasa.gov/astrobee/
2. MoveIt: https://moveit.picknik.ai/main/index.html

# Platform
Currently development is on a Apple Silicon with Docker containerization for our research and benchmarking. Better performance (and possibly less boilerplate) would be achieved with Linux
Ubuntu since Astrobee and MoveIt are all Ubuntu-native. 

# Coding Assistance
ChaptGPT 5.2

# Running the services
The stack provides a simulation platform for researching and testing micro-gravity environment Optimal Control and Deep Learning based trajectory planners and executors.
To start:
1. In the project root dir, run ./bring_up.sh to boot up the docker containers: 
    rosnet(bridge network), ros_master, astrobee_grasp, moveit
    start.sh brings up the stack and runs tests on the environment, and test that
    the servers can communicate via ROS and send and receive messages. 
    It also runs diagnostic tests of the pose, intercept, and trajectory planners. 
2. Run ./stop.sh to stop the services and tear down the stack and free any memory/volumes

# Training DGM
1. rosrun object_tracking dgm_pretrain.py _T:=2.0 _iters:=300 _batch:=192

# Work in Progress
1. Improve DGM-HJB training and intercept algorithms.
2. Ensure DGM better includes valid trajectory and valid/invalid states checking via cost function penalties, constrains 
    or a neural network prediction.
3. Improve intercept planning/metrics from proximity to contact and grasping.

# Architecture
There are mainly two(2) docker containers (astrobee and moveit) that communicate by a ros bridge container.
Object instantiation and Perception are handled in the astrobee container which uses NASA's astrobee project
to provide a free flying object mimicking drift in microgravity. Astrobee container broadcasts object location and 
attitude to its object/state node. Moveit container subscribes to the object state topics and uses that to
instantiate the planning environment. The post-perception actions happen in moveit container. 

Two(2) planning methods are implemented. Our custom Deep Galerkin Method (DGM), and MoveIt's Open Motion Planning 
library (OMPL). OMPL is great for benchmarking since it's a battle tested MoveIt internal method. OMPL provides a 
foundation for ensuring our DGM works in a simulation and real environment, and also helps to compare and benchmark DGM.

Mainly, DGM (using Hamilton-Jacobi-Method) training happens offline. During inference, we rollout using the pretrained DGM network, and check the generated
trajectory for invalid states using in a manner replicating OMPL's checks for invalid trajectories. Other checks and constraints are also applied
to ensure that the DGM trajectories are smooth and valid and would transfer well to real robots safely. A current exploration is into how to 
introduce these constraints as penalties in the DGM cost function and/or whether to train a small neural network to predict invalid/valid states to help 
with the DGM trajectory rollouts. Though DGM (with minimal training) is currently generating valid trajectories just like OMPL, it's taking longer times and tries; 
it could be improved upon with some of these validity checks baked in. 
 
Online, a free-flying object instantiated by Astrobee provides the object location and pose to the Trajectory Executor Manager,
scene planning, and intercept planning services in the moveit container. Both or either of OMPL and DGM planners are called 
to generate trajectories for intercepting the object. Intercept metrics are then computed for benchmarking DGM. 



                                    ┌──────────────────────────────┐
                                    │        Astrobee / Vision     │
                                    │   (sim or future VLM stack)  │
                                    └──────────────┬───────────────┘
                                                   │
                                                   ▼
                                      /object/state  (nav_msgs/Odometry)
                                                   │
                         ┌─────────────────────────┴─────────────────────────┐
                         │                                                   │
                         ▼                                                   ▼
     ┌─────────────────────────────────────┐              ┌─────────────────────────────────────┐
     │ object_to_planning_scene.py         │              │ trajectory_executor_manager.py      │
     │ - converts object state to MoveIt   │              │ - main orchestrator                │
     │   collision/world object            │              │ - /start_trial service             │
     │ - updates planning scene            │              │ - /run_benchmark service           │
     └─────────────────┬───────────────────┘              │ - plan → execute → evaluate        │
                       │                                  │ - retry / replan up to max attempts│
                       ▼                                  │ - writes CSV                       │
         ┌──────────────────────────────┐                 │ - publishes /benchmark/summary     │
         │ MoveIt Planning Scene        │                 └──────────────┬──────────────────────┘
         │ - world objects              │                                │
         │ - robot state                │                                │
         │ - allowed collision matrix   │                                │
         └──────────────┬───────────────┘                                │
                        │                                                │
                        │                              ┌─────────────────┴─────────────────┐
                        │                              │                                   │
                        ▼                              ▼                                   ▼
         ┌──────────────────────────────┐   ┌──────────────────────────────┐   ┌──────────────────────────────┐
         │ OMPL planner                 │   │ DGM planner                  │   │ ExecuteTrajectoryAction      │
         │ /plan_kinematic_path         │   │ /dgm/get_motion_plan         │   │ /execute_trajectory          │
         │ - MoveIt native planner      │   │ dgm_planner_node.py          │   │ - fake controller manager    │
         └──────────────┬───────────────┘   │ - HJB/DGM rollout            │   │   or future real controller  │
                        │                   │ - IK feasibility check       │   └──────────────┬───────────────┘
                        │                   │ - limit/smoothness checks    │                  │
                        │                   │ - /check_state_validity      │                  │
                        │                   │ - optional Jacobian hook     │                  │
                        │                   └──────────────┬───────────────┘                  │
                        │                                  │                                  │
                        └──────────────────────┬───────────┴───────────┬──────────────────────┘
                                               │                       │
                                               ▼                       ▼
                                  planned RobotTrajectory      executed robot motion / TF
                                               │                       │
                                               └───────────┬───────────┘
                                                           │
                                                           ▼
                                      ┌─────────────────────────────────────┐
                                      │ intercept_evaluator.py              │
                                      │ - always-on evaluator               │
                                      │ - TF: world ↔ panda_hand            │
                                      │ - TF: world ↔ object_link           │
                                      │ - publishes /intercept/eval/metrics │
                                      └─────────────────┬───────────────────┘
                                                        │
                                                        ▼
                                      ┌─────────────────────────────────────┐
                                      │ trajectory_executor_manager.py      │
                                      │ - gates evaluator metrics by trial  │
                                      │ - decides success / failure         │
                                      │ - triggers retry or trial complete  │
                                      └─────────────────────────────────────┘  


# References
TBD.