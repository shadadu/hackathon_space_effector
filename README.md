# Space Effector
An implementation of a robotic end-effector for micro-gravity environments to intercept and grasp objects. 
Gearing for grasp and move activities inside or outside space stations and modules.  

Stack: 
1. Micro-gravity free flying object inspired by NASA Astrobee: Github https://github.com/nasa/astrobee, Main site: https://www.nasa.gov/astrobee/
2. MoveIt Grasping manipulator: https://moveit.picknik.ai/main/index.html

# Platform
WSL2 + Nvidia Cuda + JAX 


# 1. Start Desktop Docker

# 2. Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi

# 3. Build with GPU JAX
cd /home/shad/Git/hackathon_space_effector
docker build -f moveit_build/Dockerfile --build-arg JAX_VARIANT=cuda12 -t moveit_image:noetic moveit_build/

# 4. Run with GPU enabled
MOVEIT_ENABLE_GPU=true JAX_VARIANT=cuda12 ./bring_up.sh


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

nvidia-smi

docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu20.04 nvidia-smi

JAX_VARIANT=cuda12 MOVEIT_ENABLE_GPU=true ./bring_up.sh

# Set these environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"

# Or use the setup script
source /path/to/setup_training_env.sh

# Training DGM
1. rosrun object_tracking dgm_training.py _T:=2.0 _iters:=300 _ft_iters:=0 _batch:=192


# Running DGM Planner
Open separate terminal windows for each command with `docker exec -it moveit bash`

`rosrun object_tracking dgm_planner_node.py _service_name:=/dgm/get_motion_plan _ik_service:=/compute_ik _group_name:=panda_arm _ee_link:=panda_hand _T:=3.0`

`rosrun object_tracking trajectory_executor_manager.py _group_name:=panda_arm _ee_link:=panda_hand _plan_service:=/dgm/get_motion_plan _execute_action:=/execute_trajectory _object_topic:=/object/state _world_frame:=world _max_attempts:=1 _loop_hz:=1.0`

`rosservice call /start_trial "{max_attempts: 1, eps_pos: 0.15, eps_ang: 1.56, eval_window_s: 5.0}"`

# Work in Progress

# Perfomance Update
Currently MoveIt's internal motion planners, OOMPL, are able to reach within proximity tolerances of the end-effector goal, while the initial basic
implementation of DGM (using a simple network) wasn't reaching the proximity tolerances. Since then, the Value Network has been updated 
to the DGM Layer recommended by the original DGM authors. Cost functions are being experimented with velocity and joint position costs before 
the next rollout or inference benchmarks. 

The Value function V(q, g, t, T) is the Euclidean distance from goal g to the current location q at time t and time limit T. 
Minimizing the value function in the subsequent time steps brings the end-effector closer to the end goal. Current solver
uses the Riccati solution of the HJB formulation to compute the control velocities u*. A neural network (DGMValueNetJAX) is trained on 
data of end-effector start and goal locations sampled from a xyz boundary region and start and intermediate valid states sampled from the robot arm 
collision free initial joint states. 

The basic HJB-DGM training is an iterative loop that computes the Value Network V, which is then used to obtain the optimal control u* via the Riccati-like 
equation u* = - Grad(V) * R^-1 * Grad(V)

running_cost --> hjb_residual_loss --> terminal_cost --> loss = hjb_residual_loss + terminal_cost --> update weights V --> Grad V --> Ricatti --> u* --> running_cost

In essence, the Neural Net is trained to generate robot arm joint velocities given current joint and EE states, and the EE goal position. The Euler method is then used to generate the trajectory by iteratively computing q_{k+1} = q_k + u_star_k * dt. Repeated computations should then bring the EE position q_k closer to g_T (within an acceptable proximity tolerance). 

Constraints such as joint velocity limits, valid states, are enforced during data generation at training and also during planing rollouts. Following Physics Informed Neural Networks(PINNs), there are two (2) loss terms that are used to train the value network: the terminal loss which penalizes deviations of the EE position from the goal position at terminal time T, and the residual or PDE loss which is the residual of the HJB equation. The residual loss thus ensures that the NN is training to solve the HJB PDE which describes the overall physics behavior of the robot arm, as occuring within the xyz domain bounds. Similar to numerical solutions of a PDE dP/dt = 0, the residual R = dP/dt, hence as the numerical or neural network training improves we expect R -> 0. And so the residual loss can be used to train a NN to approximate a PDE, in this case the HJB. Initial and Terminal Conditions(ICs and TCs), Boundary Conditions (BCs), can be added as additional terms to enable the NN to better capture the physical behaviors. In this robot manipulator solution using HJB formulation, we include the EE goal position as the Terminal Condition. So the NN trains according to loss = loss_pde + Cr * loss_terminal, where Cr is a weighting factor. During training, if necessary, some additional samples of the TC are added to the residual batches to improve training stability and convergence. 

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
6. Valin, A. (2026) Multi-Trajectory Physics-Informed Neural Networks for HJB equations with hard terminal constraints: Optimal Execution and high-dimensional LQR
6. Black et al. (2024) Pi0: A Vision-Language-Action Flow Model for General Robot Control

