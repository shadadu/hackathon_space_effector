# High-level summary of a benchmark run (e.g., 100 trials)

std_msgs/Header header

string run_id
string planner_a   # e.g. "OMPL"
string planner_b   # e.g. "DGM"

uint32 n_trials

# Success counts
uint32 success_a
uint32 success_b

# Intercept success counts (if you separate planning success vs intercept success)
uint32 intercept_success_a
uint32 intercept_success_b

# Aggregate timing (seconds)
float64 mean_plan_time_a
float64 std_plan_time_a
float64 mean_plan_time_b
float64 std_plan_time_b

# Aggregate trajectory metrics
float64 mean_path_length_a
float64 mean_path_length_b

# Aggregate intercept metrics
float64 mean_time_to_intercept_a
float64 mean_time_to_intercept_b

# Notes / diagnostics
string notes