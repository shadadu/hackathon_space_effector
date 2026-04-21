# High-level benchmark summary (one run)
string run_id
int32 trials
int32 seed

# Services used
string ompl_service
string dgm_service
string validity_service
string jacobian_service

# Workspace bounds and tolerances
float64 x_min
float64 x_max
float64 y_min
float64 y_max
float64 z_min
float64 z_max
float64 pos_tol

# Planning configuration
float64 allowed_planning_time
int32 num_planning_attempts
float64 vel_scale
float64 acc_scale
int32 validity_check_stride
bool snapshot_jacobian

# Success rates
int32 ompl_success
int32 dgm_success
float64 ompl_success_rate
float64 dgm_success_rate

# "All-valid" rates (passes /check_state_validity on sampled waypoints + time monotone + no joint limit violation)
int32 ompl_all_valid
int32 dgm_all_valid
float64 ompl_all_valid_rate
float64 dgm_all_valid_rate

# Timing stats (planning_time reported by MoveIt)
float64 ompl_plan_mean
float64 ompl_plan_median
float64 ompl_plan_p90
float64 dgm_plan_mean
float64 dgm_plan_median
float64 dgm_plan_p90

# Validity stats (fraction of sampled waypoints valid)
float64 ompl_valid_frac_mean
float64 ompl_valid_frac_median
float64 ompl_valid_frac_p90
float64 dgm_valid_frac_mean
float64 dgm_valid_frac_median
float64 dgm_valid_frac_p90

# Smoothness penalties (finite-diff)
float64 ompl_acc_pen_mean
float64 ompl_jerk_pen_mean
float64 dgm_acc_pen_mean
float64 dgm_jerk_pen_mean

# Output files
string csv_path