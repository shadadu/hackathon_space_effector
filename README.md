# Space Effector
An implementation of a robotic end-effector for micro-gravity environments to intercept and grasp objects. 
Gearing for grasp and move activities inside or outside space stations and modules.  

Stack: 
1. NASA Astrobee: Github https://github.com/nasa/astrobee, Main site: https://www.nasa.gov/astrobee/
2. MoveIt: https://moveit.picknik.ai/main/index.html

# Platform
Original platform was MacOS with docker. Robotics platforms such as MoveIt, Gazebo, ROS, don't play very well on MacOS; they're are more functional on Linux/Ubuntu. Migration to Ubuntu VM + Docker is nearly complete.

Run the below command to launch the perception view in Gazebo. This displays the object without the MoveIt Panda arm.

`
xhost +local:docker

ASTROBEE_ENABLE_X11=true ASTROBEE_LAUNCH='roslaunch astrobee_grasp gazebo_perception.launch gui:=true' ./bring_up.sh
`
# Coding Assistance
ChaptGPT 5.2+, Codex

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

# Perfomance Update
Currently MoveIt's internal motion planners, OOMPL, are able to reach within proximity tolerances of the end-effector goal, while the initial basic
implementation of DGM (using a simple network) wasn't reaching the proximity tolerances. Since then, the Value Network has been updated 
to the DGM Layer recommended by the original DGM authors. Cost functions are being experimented with velocity and joint position costs before 
the next rollout or inference benchmarks. 

The Value function V(q, g, t, T) is the Euclidean distance from goal g to the current location q at time t and time limit T. 
Minimizing the value function in the subsequent time steps brings the end-effector closer to the end goal. Current solver
uses the Riccati solution of the HJB formulation to compute the control velocities u*. A neural network (DGMValueNet) is trained on 
data of end-effector goal locations sampled from a xyz boundary region and start and intermediate valid states sampled from the robot arm 
collision free initial joint states. In essence, the Neural Net is trained to generate robot arm joint state trajectories 
for given end-effector goal positions.

The basic HJB-DGM training is an iterative loop that computes the Value Network V, which is then used to obtain the optimal control u* via the Riccati-like 
equation u* = - Grad(V) * R^-1 * Grad(V)

running_cost --> hjb_residual_loss --> terminal_cost --> loss = hjb_residual_loss + terminal_cost --> update weights V --> Grad V --> Ricatti --> u* --> running_cost

More advanced implementations(see references) essentially replace the Riccati equation for estimating u* with a neural network and train both the 
Value Net and the Control Net(u*) with an Actor-Critic RL method. 

Currently, with a well calibrated weights of the terminal and residual losses, we are getting good convergence of the total loss function, though the 
HJB residual part of the loss seems unstable, the control loss converges smoothly. Smaller learning rates also improve 
the HJB residual's convergence relative to the terminal loss, but there is still room for improvement.
Experiments are to be run to see how the relative convergence of terminal and residual losses affects overall trajectory planning
performance. However, based on the recommendations in the literature, the Actor-Critic method with two(2) neural nets seem to be better solution, 
and the next major upgrade would be that, after benchmarking the Value Net and Riccati-like method.

Constraints such as joint velocity limits, valid states, are enforced during data generation at training time. Because the residual 
only reduces to about 20% of its value before plateauing, it is likely that HJB formulation is leaving out significant robotic
behavior. The direction of improvement going forward is to improve the loss functions and the Initial and Boundary conditions, and 
constraints to better coverage. Physics Informed Neural Network (PINN) formulations usually include constraint, IC and BC and residual
losses in the total loss function, so that is direction to explore.

# Architecture
There are mainly two(2) docker containers -- astrobee(emulator) and moveit that communicate by a ros bridge container.
Object instantiation and Perception are handled in the astrobee container which uses NASA's astrobee project
to provide a free flying object mimicking drift in microgravity. Astrobee container broadcasts object location and 
attitude to its object/state node. Moveit container subscribes to the object state topics and uses that to
instantiate the planning environment. The post-perception actions happen in moveit container. 

Two(2) planning methods are implemented. Our custom Deep Galerkin Method (DGM), and MoveIt's Open Motion Planning 
library (OMPL). OMPL is great for benchmarking since it's a battle tested MoveIt internal method. OMPL provides a 
foundation for ensuring our DGM works in a simulation and real environment, and also helps to compare and benchmark DGM.

Mainly, DGM (using Hamilton-Jacobi-Method) training happens offline. During inference, we rollout using the pretrained DGM network, and check the generated
trajectory for invalid states using in a manner replicating OMPL's checks for invalid trajectories. Other checks and constraints are also applied
to ensure that the DGM trajectories are smooth and valid and would transfer well to real robots safely. A current exploration is how to 
introduce these constraints as penalties in the cost functions and/or whether to train a small neural network to predict invalid/valid states to help 
with the DGM trajectory rollouts. Though DGM (with minimal training) is currently generating valid trajectories just like OMPL, it's taking longer times and tries; 
it could be improved upon with some of these validity checks baked in. 
 
Online, a free-flying object instantiated by Astrobee emulator provides the object location and pose to the Trajectory Executor Manager,
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
1. A. Borovykh et al. (2022) Data-driven initialization of deep learning solvers for Hamilton-Jacobi-Bellman PDEs
2. A. Al Aradi et al. (2018) Solving Nonlinear and High-Dimensional Partial Differential Equations via Deep Learning
3. Beard R. et al. (1997) Galerkin Approximations of the Generalized Hamilton-Jacobi-Bellman Equation
4. Detorakis, G. I (2024) Practical Aspects on Solving Differential Equations using Deep Learning: A primer
5. Sirignano, J. et al. (2018) DGM: A deep learning algorithm for solving partial differential equations

