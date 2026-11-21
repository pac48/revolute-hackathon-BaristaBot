# The whole MuJoCo state of a simulation: qpos and qvel as MuJoCo orders them (the model's nq and nv
# values; a free joint is 7 positions, x y z qw qx qy qz, and 6 velocities, linear then angular, world frame).
# Published by //src/ros2_control_sim/mit_system_interface on every read() of the hardware interface.

# The simulation time of the state (header.stamp: the controller manager's time of the read()).
std_msgs/Header header
float64 sim_time
float64[] qpos
float64[] qvel
