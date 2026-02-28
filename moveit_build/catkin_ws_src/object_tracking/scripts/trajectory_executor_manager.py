#!/usr/bin/env python3
import time
import uuid
import threading
import math
import rospy
import actionlib

from nav_msgs.msg import Odometry
from std_msgs.msg import String

from moveit_msgs.srv import GetMotionPlan, GetMotionPlanRequest
from moveit_msgs.msg import MotionPlanRequest, MoveItErrorCodes
from moveit_msgs.msg import ExecuteTrajectoryAction, ExecuteTrajectoryGoal

from object_tracking.msg import InterceptMetrics
from object_tracking.srv import StartTrial, StartTrialResponse
from object_tracking.srv import RunBenchmark, RunBenchmarkResponse

from geometry_msgs.msg import PoseStamped, Vector3
from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, RobotState
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState




def make_robot_state_from_joint_dict(joint_dict):
    js = JointState()
    js.name = list(joint_dict.keys())
    js.position = [joint_dict[n] for n in js.name]
    rs = RobotState()
    rs.joint_state = js
    return rs


def panda_extended_open_start_state():
    joints = {
        "panda_joint1": 0.0,
        "panda_joint2": 0.0,
        "panda_joint3": 0.0,
        "panda_joint4": 0.0,
        "panda_joint5": 0.0,
        "panda_joint6": 1.571,
        "panda_joint7": 0.785,
        "panda_finger_joint1": 0.035,
        "panda_finger_joint2": 0.035,
    }
    return make_robot_state_from_joint_dict(joints)


def make_goal_constraints_from_pose(goal_pose_stamped: PoseStamped, ee_link: str,
                                    pos_tol=0.02, ang_tol=0.35):
    c = Constraints()

    pc = PositionConstraint()
    pc.header = goal_pose_stamped.header
    pc.link_name = ee_link
    pc.target_point_offset = Vector3(0.0, 0.0, 0.0)

    box = SolidPrimitive()
    box.type = SolidPrimitive.BOX
    box.dimensions = [pos_tol, pos_tol, pos_tol]
    pc.constraint_region.primitives.append(box)
    pc.constraint_region.primitive_poses.append(goal_pose_stamped.pose)
    pc.weight = 1.0

    oc = OrientationConstraint()
    oc.header = goal_pose_stamped.header
    oc.link_name = ee_link
    oc.orientation = goal_pose_stamped.pose.orientation
    oc.absolute_x_axis_tolerance = ang_tol
    oc.absolute_y_axis_tolerance = ang_tol
    oc.absolute_z_axis_tolerance = ang_tol
    oc.weight = 1.0

    c.position_constraints.append(pc)
    c.orientation_constraints.append(oc)
    return c


class TrajectoryExecutorManager:
    """
    Service-driven trial runner:
      - /start_trial: run one trial (async, returns immediately)
      - /run_benchmark: run N trials (async)
    Always-on evaluator publishes /intercept/eval/metrics; we gate it per active trial.
    """

    def __init__(self):
        # Inputs
        self.object_topic = rospy.get_param("~object_topic", "/object/state")
        self.metrics_topic = rospy.get_param("~metrics_topic", "/intercept/eval/metrics")

        # Frames / group
        self.world_frame = rospy.get_param("~world_frame", "world")
        self.group_name = rospy.get_param("~group_name", "panda_arm")
        self.ee_link = rospy.get_param("~ee_link", "panda_hand")

        # Default planner service (may be overridden per trial)
        self.default_plan_service = rospy.get_param("~plan_service", "/plan_kinematic_path")

        # Execution action
        self.exec_action = rospy.get_param("~execute_action", "/execute_trajectory")

        # Defaults (overridable per trial)
        self.max_attempts_default = int(rospy.get_param("~max_attempts", 5))
        self.eval_window_default = float(rospy.get_param("~eval_window_s", 2.5))
        self.eps_pos_default = float(rospy.get_param("~eps_pos", 0.03))
        self.eps_ang_default = float(rospy.get_param("~eps_ang", 0.35))

        # Planning params
        self.allowed_planning_time = float(rospy.get_param("~allowed_planning_time", 2.0))
        self.num_planning_attempts = int(rospy.get_param("~num_planning_attempts", 3))
        self.vel_scale = float(rospy.get_param("~vel_scale", 0.3))
        self.acc_scale = float(rospy.get_param("~acc_scale", 0.3))

        # Freshness and loop
        self.object_max_age_s = float(rospy.get_param("~object_max_age_s", 0.25))
        self.loop_hz = float(rospy.get_param("~loop_hz", 10.0))

        # State
        self.last_odom = None
        self.last_metrics = None

        self.start_state = panda_extended_open_start_state()

        # Status
        self.status_pub = rospy.Publisher("/trajectory_executor/status", String, queue_size=10)

        # Subscribers
        rospy.Subscriber(self.object_topic, Odometry, self.cb_object, queue_size=1)
        rospy.Subscriber(self.metrics_topic, InterceptMetrics, self.cb_metrics, queue_size=50)

        # ExecuteTrajectoryAction
        self.exec_client = actionlib.SimpleActionClient(self.exec_action, ExecuteTrajectoryAction)
        rospy.loginfo("Waiting for ExecuteTrajectoryAction: %s", self.exec_action)
        if not self.exec_client.wait_for_server(rospy.Duration(60.0)):
            raise RuntimeError(f"ExecuteTrajectoryAction not available: {self.exec_action}")

        # Services
        self.srv_start = rospy.Service("/start_trial", StartTrial, self.on_start_trial)
        self.srv_bench = rospy.Service("/run_benchmark", RunBenchmark, self.on_run_benchmark)

        # Threading
        self.run_lock = threading.Lock()
        self.worker_thread = None
        self.stop_flag = False

        # Active-trial gating
        self.active = None  # dict or None

        # Timer just watches + handles state machine
        self.timer = rospy.Timer(rospy.Duration(1.0 / self.loop_hz), self.on_timer)

        rospy.loginfo("TrajectoryExecutorManager ready: /start_trial and /run_benchmark available.")

    # ----------------- Callbacks -----------------
    def cb_object(self, msg: Odometry):
        self.last_odom = msg

    def cb_metrics(self, msg: InterceptMetrics):
        self.last_metrics = msg

    def publish_status(self, s: str):
        self.status_pub.publish(String(data=s))

    # ----------------- Helpers -----------------
    def object_fresh(self):
        if self.last_odom is None:
            return False
        age = (rospy.Time.now() - self.last_odom.header.stamp).to_sec()
        return (age >= 0.0) and (age <= self.object_max_age_s)

    def pick_goal_pose(self, odom: Odometry) -> PoseStamped:
        # Minimal V1: go to object's current pose (you can replace with intercept selection logic)
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = odom.header.frame_id or self.world_frame
        goal.pose.position = odom.pose.pose.position
        goal.pose.orientation = odom.pose.pose.orientation
        return goal

    def build_mpr(self, goal: PoseStamped, eps_pos: float, eps_ang: float) -> MotionPlanRequest:
        mpr = MotionPlanRequest()
        mpr.group_name = self.group_name
        mpr.num_planning_attempts = self.num_planning_attempts
        mpr.allowed_planning_time = self.allowed_planning_time
        mpr.max_velocity_scaling_factor = self.vel_scale
        mpr.max_acceleration_scaling_factor = self.acc_scale
        mpr.start_state = self.start_state
        mpr.goal_constraints = [make_goal_constraints_from_pose(goal, self.ee_link,
                                                                pos_tol=eps_pos,
                                                                ang_tol=eps_ang)]
        return mpr

    def call_planner(self, service_name: str, mpr: MotionPlanRequest):
        rospy.wait_for_service(service_name, timeout=30.0)
        proxy = rospy.ServiceProxy(service_name, GetMotionPlan)
        req = GetMotionPlanRequest()
        req.motion_plan_request = mpr
        t0 = time.time()
        resp = proxy(req).motion_plan_response
        dt = time.time() - t0
        return resp, dt

    def execute_trajectory(self, traj):
        g = ExecuteTrajectoryGoal()
        g.trajectory = traj
        self.exec_client.send_goal(g)

    def cancel_execution(self):
        try:
            self.exec_client.cancel_goal()
        except Exception:
            pass

    # ----------------- Trial engine -----------------
    def _new_trial_id(self, given: str):
        if given and given.strip():
            return given.strip()
        return f"trial_{uuid.uuid4().hex[:8]}"

    def _start_trial_internal(self, planner_service: str, max_attempts: int,
                              eval_window_s: float, eps_pos: float, eps_ang: float,
                              trial_id: str):
        # Prepare active gating struct
        self.active = {
            "trial_id": trial_id,
            "planner_service": planner_service,
            "attempt_idx": 0,
            "max_attempts": max_attempts,
            "eval_window_s": eval_window_s,
            "eps_pos": eps_pos,
            "eps_ang": eps_ang,
            "attempt_start_t": None,
            "attempt_deadline_t": None,
            "min_dist": float("inf"),
            "min_ang": float("inf"),
            "planner_time_s": None,
        }
        rospy.loginfo("START trial=%s planner=%s max_attempts=%d eps_pos=%.3f eps_ang=%.3f window=%.2f",
                      trial_id, planner_service, max_attempts, eps_pos, eps_ang, eval_window_s)

    def _finish_trial(self, success: bool, reason: str):
        if self.active is None:
            return
        tid = self.active["trial_id"]
        planner = self.active["planner_service"]
        k = self.active["attempt_idx"]
        md = self.active["min_dist"]
        ma = self.active["min_ang"]
        pt = self.active["planner_time_s"]
        rospy.loginfo(
            "END trial=%s success=%s reason=%s planner=%s attempt=%d min_dist=%.4f min_ang=%.4f plan_time=%.4f",
            tid, str(success), reason, planner, k, md, ma, (pt if pt is not None else -1.0))
        self.publish_status(f"TRIAL_END {tid} success={success} reason={reason} min_dist={md:.4f}")
        self.active = None

    def _attempt_step(self):
        """
        One tick of the trial state machine.
        Called from timer (single-threaded ROS callback).
        """
        if self.active is None:
            return

        # Update mins while active
        if self.last_metrics is not None:
            self.active["min_dist"] = min(self.active["min_dist"], float(self.last_metrics.distance_m))
            self.active["min_ang"] = min(self.active["min_ang"], float(self.last_metrics.angle_rad))

        # If attempt not started yet, plan+execute
        if self.active["attempt_start_t"] is None:
            if not self.object_fresh():
                self.publish_status("WAIT_OBJECT_FRESH")
                return

            goal = self.pick_goal_pose(self.last_odom)
            mpr = self.build_mpr(goal, self.active["eps_pos"], self.active["eps_ang"])

            self.publish_status(f"PLANNING {self.active['trial_id']} attempt={self.active['attempt_idx']}")
            try:
                plan_resp, plan_dt = self.call_planner(self.active["planner_service"], mpr)
            except Exception as e:
                rospy.logwarn("Planner call failed: %s", str(e))
                self.active["attempt_idx"] += 1
                if self.active["attempt_idx"] >= self.active["max_attempts"]:
                    self._finish_trial(False, "planner_call_failed")
                return

            self.active["planner_time_s"] = plan_dt

            if plan_resp.error_code.val != MoveItErrorCodes.SUCCESS:
                rospy.logwarn("Planning failed error_code=%d", plan_resp.error_code.val)
                self.active["attempt_idx"] += 1
                if self.active["attempt_idx"] >= self.active["max_attempts"]:
                    self._finish_trial(False, f"plan_fail_{plan_resp.error_code.val}")
                return

            # Execute + set gating window
            now = rospy.Time.now().to_sec()
            self.active["attempt_start_t"] = now
            self.active["attempt_deadline_t"] = now + self.active["eval_window_s"]
            self.publish_status(f"EXECUTING {self.active['trial_id']} attempt={self.active['attempt_idx']}")
            self.execute_trajectory(plan_resp.trajectory)
            return

        # Attempt running: check verdict inside window
        now = rospy.Time.now().to_sec()

        # Success condition (gated)
        if self.last_metrics is not None:
            d_ok = (self.last_metrics.distance_m <= self.active["eps_pos"])
            a_ok = (self.last_metrics.angle_rad <= self.active["eps_ang"])
            if d_ok and a_ok:
                self._finish_trial(True, "within_tolerance")
                return

        # Fail if exceeded window
        if now > self.active["attempt_deadline_t"]:
            self.cancel_execution()
            self.active["attempt_idx"] += 1
            if self.active["attempt_idx"] >= self.active["max_attempts"]:
                self._finish_trial(False, "timeout_window")
                return

            # Reset for replanning next tick
            self.active["attempt_start_t"] = None
            self.active["attempt_deadline_t"] = None
            self.publish_status(f"REPLAN {self.active['trial_id']} next_attempt={self.active['attempt_idx']}")
            return

    # ----------------- Services -----------------
    def on_start_trial(self, req):
        with self.run_lock:
            if self.active is not None:
                return StartTrialResponse(
                    accepted=False,
                    message="Busy: a trial is already running",
                    active_trial_id=self.active["trial_id"],
                )

            planner = req.planner_service.strip() if req.planner_service else self.default_plan_service
            max_attempts = int(req.max_attempts) if req.max_attempts > 0 else self.max_attempts_default
            window = float(req.eval_window_s) if req.eval_window_s > 0 else self.eval_window_default
            eps_pos = float(req.eps_pos) if req.eps_pos > 0 else self.eps_pos_default
            eps_ang = float(req.eps_ang) if req.eps_ang > 0 else self.eps_ang_default
            tid = self._new_trial_id(req.trial_id)

            self._start_trial_internal(planner, max_attempts, window, eps_pos, eps_ang, tid)

            return StartTrialResponse(
                accepted=True,
                message="Trial accepted and started",
                active_trial_id=tid,
            )

    def on_run_benchmark(self, req):
        with self.run_lock:
            if self.worker_thread is not None and self.worker_thread.is_alive():
                return RunBenchmarkResponse(False, "Benchmark already running")

            if req.num_trials <= 0:
                return RunBenchmarkResponse(False, "num_trials must be > 0")

            planner_a = req.planner_a.strip() if req.planner_a else "/plan_kinematic_path"
            planner_b = req.planner_b.strip() if req.planner_b else ""
            alternate = bool(req.alternate)

            max_attempts = int(req.max_attempts) if req.max_attempts > 0 else self.max_attempts_default
            window = float(req.eval_window_s) if req.eval_window_s > 0 else self.eval_window_default
            eps_pos = float(req.eps_pos) if req.eps_pos > 0 else self.eps_pos_default
            eps_ang = float(req.eps_ang) if req.eps_ang > 0 else self.eps_ang_default

            # Start async worker thread that sequentially triggers trials
            self.stop_flag = False
            self.worker_thread = threading.Thread(
                target=self._benchmark_worker,
                args=(req.num_trials, planner_a, planner_b, alternate, max_attempts, window, eps_pos, eps_ang),
                daemon=True
            )
            self.worker_thread.start()
            return RunBenchmarkResponse(True, f"Benchmark started for {req.num_trials} trials")

    def _benchmark_worker(self, n, planner_a, planner_b, alternate, max_attempts, window, eps_pos, eps_ang):
        rospy.loginfo("BENCHMARK worker started: n=%d alternate=%s A=%s B=%s",
                      n, str(alternate), planner_a, planner_b)

        # Simple stats in logs (you can extend to CSV later)
        results = []

        for i in range(n):
            if self.stop_flag:
                break

            # Select planner
            planner = planner_a
            if alternate and planner_b:
                planner = planner_a if (i % 2 == 0) else planner_b

            tid = f"bench_{i:03d}_{'A' if planner == planner_a else 'B'}_{uuid.uuid4().hex[:6]}"

            # Wait until manager is free
            while not rospy.is_shutdown():
                with self.run_lock:
                    busy = (self.active is not None)
                    if not busy:
                        self._start_trial_internal(planner, max_attempts, window, eps_pos, eps_ang, tid)
                        break
                time.sleep(0.05)

            # Wait for trial to complete
            while not rospy.is_shutdown():
                with self.run_lock:
                    done = (self.active is None) or (self.active and self.active.get("trial_id") != tid)
                # Above is conservative; simplest: wait until active becomes None
                if self.active is None:
                    break
                time.sleep(0.05)

            # No direct return object; parse via logs for now.
            results.append((tid, planner))
            rospy.loginfo("BENCHMARK progress: %d/%d done", i + 1, n)

        rospy.loginfo("BENCHMARK worker finished. trials_completed=%d", len(results))

    # ----------------- Timer -----------------
    def on_timer(self, _evt):
        # Single tick of the trial engine (if active)
        try:
            self._attempt_step()
        except Exception as e:
            rospy.logerr("Trial engine exception: %s", str(e))
            # Fail-safe: cancel and drop trial
            self.cancel_execution()
            self._finish_trial(False, "exception")

    def decode(code):
        for k, v in MoveItErrorCodes.__dict__.items():
            if isinstance(v, int) and v == code:
                return k
        return str(code)

def main():
    rospy.init_node("trajectory_executor_manager")
    TrajectoryExecutorManager()
    rospy.spin()


if __name__ == "__main__":
    main()
