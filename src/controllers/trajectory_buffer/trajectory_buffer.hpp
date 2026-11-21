// A buffer of timestamped joint position targets, smoothed into a C2 trajectory within joint limits.
//
// Each joint keeps its own list of (time, position) targets. A command can name any subset of the
// joints (NaN = not commanded) and each commanded joint's targets replace its targets from the
// command's first time on: a target for joint j at 1.5 s deletes j's targets at and after 1.5 s
// (e.g. one at 3.0 s) and is added after the earlier ones. So a new trajectory preempts the
// remainder of the old one, as MPC needs, and the joints a command leaves out keep theirs. A
// joint holds its last target (zero velocity) once it has passed, until a command replaces it.
// Targets that are due are dropped from every joint at each set_commands().
//
// Options::latency delays every command by a fixed time: a target for `t` is placed at
// t + latency. Streamed commands then overlap: the targets of the last `latency` seconds of
// commands are still ahead when a new one arrives, and the fit smooths over all of them.
//
// Every set_commands() refits the commanded joints' targets from `now` on with a piecewise quintic
// (a quintic Hermite segment between consecutive knots; position, velocity and acceleration are
// the unknowns at each knot), minimizing
//
//   w_p * sum_k (p(t_k) - y_k)^2  +  w_v * integral v^2 dt  +  w_a * integral a^2 dt
//
// subject to
//   - the first knot at `now` taking the trajectory's state there (position, velocity and
//     acceleration), so a refit or preemption never jumps;
//   - the last target being reached exactly with zero acceleration, and with the command's final
//     velocity for the joint if it gives one (0: arrive at rest, then hold smoothly). Without one
//     the final velocity is free (a natural end): streamed targets are followed at the speed they
//     imply, and the joint's velocity steps to zero when it reaches its last target and holds;
//   - lower <= p(t) <= upper (Options::lower/upper) everywhere: at the knots and limit_samples points
//     of each segment, and at the extrema of each segment where a solution leaves them (re-solving
//     with them added). Targets are clamped into the limits first.
// A free knot (no target) splits the first segment, so even a one-target command leaves the fit
// something to choose (e.g. how to brake before a limit).
//
// Without limits the problem is a block-tridiagonal least-squares system, solved by a banded
// Cholesky factorization in O(points). If that solution leaves the limits, the inequality-
// constrained QP is solved by a primal-dual interior-point method (Mehrotra predictor-corrector)
// whose Newton systems keep the same band, so each iteration is O(points) too. All storage is
// allocated by the constructor: set_commands() and query_command() don't allocate, so both can run
// in a real-time loop.
//
// Times are seconds on any clock the caller picks; keep them near zero (e.g. since activation)
// rather than since the epoch, for precision. Not thread-safe: call from one thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace trajectory_buffer
{

/// Costs of the fit. Only their ratios matter. position must be > 0 and velocity or acceleration
/// > 0. A larger acceleration (or velocity) weight smooths more and follows the targets less: with
/// targets every h seconds the fit passes frequencies below about (position / (h * acceleration))^(1/4)
/// rad/s.
struct Weights
{
  double position = 1.0;      ///< per target: squared position error (rad^2)
  double velocity = 0.0;      ///< integral of the squared velocity
  double acceleration = 1e-5; ///< integral of the squared acceleration
};

struct Options
{
  std::size_t num_joints = 0;
  /// The most targets a joint holds; a command beyond that is truncated (Status::kTruncated).
  std::size_t capacity = 256;
  Weights weights;
  /// Shortest time between knots (s). Targets closer to the previous knot (or to `now`) are moved
  /// to that distance; a command whose targets are all at or before `now` keeps only its last one,
  /// at now + min_interval.
  double min_interval = 1e-3;
  /// Added to every commanded time (s), >= 0.
  double latency = 0.0;
  /// Position limits per joint (empty: none; -inf / inf: none on that side).
  std::vector<double> lower, upper;
  /// Points per segment where the limits are always enforced, besides the knots (>= 1). Wherever
  /// else the solution leaves them (found at each segment's extrema), rows are added and it is re-solved.
  std::size_t limit_samples = 1;
};

enum class Status
{
  kOk,
  kTruncated,      ///< applied, but some joint's targets were cut at the capacity
  kInvalid,        ///< ignored: sizes don't match, times aren't finite and increasing, or a value is infinite
  kLimitsViolated, ///< applied, but some joint's trajectory could not be kept within its limits
};

class TrajectoryBuffer
{
public:
  /// Throws std::invalid_argument for bad options.
  explicit TrajectoryBuffer(const Options & options);

  /// Hold `positions` (one per joint) from `time` on, with no targets.
  void reset(double time, std::span<const double> positions);

  /// Add targets. times: the targets' times (before the latency), strictly increasing; positions:
  /// row-major, a row of num_joints per time, NaN where a joint isn't commanded. Commanded joints
  /// are refit from `now`; the other joints' past knots are dropped.
  /// final_velocities: empty, or one per joint: the velocity at its last target (NaN: free).
  Status set_commands(
    double now, std::span<const double> times, std::span<const double> positions,
    std::span<const double> final_velocities = {});

  /// The trajectory at `time`: one value per joint in each output.
  void query_command(
    double time, std::span<double> positions, std::span<double> velocities,
    std::span<double> accelerations) const;

  /// The targets at `time`, before the fit: one value per joint, looked up in the joint's timeline
  /// of targets (at their buffer times, i.e. with the latency, clamped into the limits): linear
  /// between the targets either side of `time`, held at the last one after it. The timeline keeps
  /// the last target at or before the last set_commands()'s `now` and every target after it (a
  /// command replaces them from its first time on, like the knots). query_target - query_command's
  /// position is the fit's deviation from the targets.
  void query_target(double time, std::span<double> positions) const;

  /// Take effect at the next set_commands(). Throw std::invalid_argument for bad values.
  void set_weights(const Weights & weights);
  void set_latency(double latency);
  const Weights & weights() const { return options_.weights; }
  double latency() const { return options_.latency; }

  std::size_t num_joints() const { return options_.num_joints; }
  /// When joint's last target is reached (from then on it holds).
  double end_time(std::size_t joint) const;
  /// The latest end_time() of all joints.
  double end_time() const;
  /// Knots of a joint's trajectory: the start (`now` of its last fit), a free knot, its targets.
  std::size_t num_knots(std::size_t joint) const { return joints_[joint].size; }
  /// Targets a joint still has (past ones are dropped at set_commands()).
  std::size_t num_targets(std::size_t joint) const;
  /// Interior-point iterations of the last set_commands() (0: no limit was active).
  std::size_t qp_iterations() const { return qp_iterations_; }

private:
  struct State
  {
    double position = 0.0, velocity = 0.0, acceleration = 0.0;
  };
  /// Knot 0 is where the last fit started; the others are targets (y; NaN for the free knot) and
  /// the fitted state at them.
  struct Joint
  {
    std::size_t size = 0;
    double final_velocity = 0.0;  // at the last knot; NaN: free
    std::vector<double> time, target, position, velocity, acceleration;
    // The targets as commanded, for query_target: the last one at or before `now` and the ones
    // after it (the knots drop the first and start from the fit's state instead).
    std::size_t timeline_size = 0;
    std::vector<double> timeline_time, timeline_target;
  };

  /// Drop the timeline's targets before its last one at or before `now`.
  static void prune_timeline(Joint & joint, double now);
  /// Append target y at t (after the timeline's last), replacing that one if both are at or before
  /// `now`. False if the timeline is full.
  static bool append_timeline(Joint & joint, double now, double t, double y);

  State evaluate(const Joint & joint, double time) const;
  void prune(Joint & joint, double now) const;
  /// False if the limits could not be met.
  bool fit(Joint & joint, double lower, double upper);
  /// The inequality-constrained QP over n unknowns and the first m rows, from x_.
  bool solve_qp(std::size_t n, std::size_t m);
  /// The QP over m rows, solved on a working set of them.
  bool solve_limited(std::size_t n, std::size_t m);
  /// C_r x - d_r for x_.
  double row_value(std::size_t r) const;
  void swap_rows(std::size_t a, std::size_t b);

  Options options_;
  std::vector<Joint> joints_;
  // Fit workspace, sized for capacity targets: the lower band of the system matrix (hessian_: as
  // assembled; band_: factored), its right side, the solution.
  std::vector<double> hessian_, band_, rhs_, x_;
  // Limit rows: row r is sum_l coef[6 r + l] x[index[6 r + l]] <= bound[r] (index -1: no term).
  std::vector<double> row_coef_, row_bound_;
  std::vector<std::int64_t> row_index_;
  // Interior-point vectors (rows: slack, multiplier, their steps, residual; unknowns: step, residual).
  std::vector<double> slack_, dual_, dslack_, ddual_, rp_, rc_, dx_, rd_;
  std::size_t qp_iterations_ = 0;
  // Per-segment cost matrices on the scaled knot values (p0, v0 h, a0 h^2, p1, v1 h, a1 h^2) of a unit segment.
  double velocity_cost_[6][6], acceleration_cost_[6][6];
};

}  // namespace trajectory_buffer
