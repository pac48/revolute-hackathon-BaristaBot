#include "trajectory_buffer.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

// Counts heap allocations, to check that the buffer doesn't allocate after construction.
static std::atomic<long> g_allocations{0};
void * operator new(std::size_t size)
{
  ++g_allocations;
  if (void * p = std::malloc(size ? size : 1)) return p;
  throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }

namespace
{
using trajectory_buffer::Options;
using trajectory_buffer::Status;
using trajectory_buffer::TrajectoryBuffer;
using trajectory_buffer::Weights;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

struct Sample
{
  std::vector<double> p, v, a;
};

Sample query(const TrajectoryBuffer & buffer, double t)
{
  Sample s{std::vector<double>(buffer.num_joints()), std::vector<double>(buffer.num_joints()),
           std::vector<double>(buffer.num_joints())};
  buffer.query_command(t, s.p, s.v, s.a);
  return s;
}

TrajectoryBuffer make(
  std::size_t joints, Weights weights = {}, std::size_t capacity = 256, double latency = 0.0,
  std::vector<double> lower = {}, std::vector<double> upper = {})
{
  Options options;
  options.num_joints = joints;
  options.capacity = capacity;
  options.weights = weights;
  options.latency = latency;
  options.lower = std::move(lower);
  options.upper = std::move(upper);
  TrajectoryBuffer buffer(options);
  buffer.reset(0.0, std::vector<double>(joints, 0.0));
  return buffer;
}

TEST(TrajectoryBuffer, HoldsAfterReset)
{
  auto buffer = make(2);
  buffer.reset(0.0, std::vector<double>{0.5, -1.0});
  const auto s = query(buffer, 10.0);
  EXPECT_EQ(s.p, (std::vector<double>{0.5, -1.0}));
  EXPECT_EQ(s.v, (std::vector<double>{0.0, 0.0}));
  EXPECT_EQ(s.a, (std::vector<double>{0.0, 0.0}));
}

TEST(TrajectoryBuffer, ReachesTheLastTargetAtRestAndHolds)
{
  auto buffer = make(1);
  const std::vector<double> times{1.0}, positions{2.0}, rest{0.0};
  ASSERT_EQ(buffer.set_commands(0.0, times, positions, rest), Status::kOk);
  const auto start = query(buffer, 0.0);
  EXPECT_NEAR(start.p[0], 0.0, 1e-12);
  EXPECT_NEAR(start.v[0], 0.0, 1e-12);
  const auto mid = query(buffer, 0.5);
  EXPECT_NEAR(mid.p[0], 1.0, 1e-12);  // a rest-to-rest quintic is symmetric
  EXPECT_GT(mid.v[0], 0.0);
  const auto end = query(buffer, 1.0 - 1e-9);
  EXPECT_NEAR(end.p[0], 2.0, 1e-8);
  EXPECT_NEAR(end.v[0], 0.0, 1e-6);
  const auto after = query(buffer, 5.0);
  EXPECT_EQ(after.p[0], 2.0);
  EXPECT_EQ(after.v[0], 0.0);
  EXPECT_DOUBLE_EQ(buffer.end_time(), 1.0);
}

// The example of the design: j0 has targets at 1.0 and 3.0; a command at 1.5 keeps 1.0 and drops 3.0.
// Without a final velocity the end is free (zero acceleration): a positions-only target is reached
// moving, and the joint then holds it.
TEST(TrajectoryBuffer, FreeEndArrivesMovingThenHolds)
{
  auto buffer = make(1);
  ASSERT_EQ(buffer.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{1.0}), Status::kOk);
  const auto end = query(buffer, 1.0 - 1e-9);
  EXPECT_NEAR(end.p[0], 1.0, 1e-8);
  EXPECT_GT(end.v[0], 0.5);
  EXPECT_NEAR(end.a[0], 0.0, 1e-5);
  const auto after = query(buffer, 1.5);
  EXPECT_EQ(after.p[0], 1.0);
  EXPECT_EQ(after.v[0], 0.0);
}

// A stream: every 5 ms one target h ahead on a ramp at 1 rad/s (the earlier targets still ahead
// stay; with h = 0.02 one lands within floating-point error of `now` every 4 cycles, which must not
// make a degenerate segment). The joint moves at the ramp's speed.
TEST(TrajectoryBuffer, StreamedTargetsAreFollowedAtTheirSpeed)
{
  const auto speed_range = [](bool rest, double horizon)
  {
    auto buffer = make(1);
    double low = 1e9, high = -1e9;
    for (int cycle = 0; cycle < 400; ++cycle)
    {
      const double now = 0.005 * cycle;
      const std::vector<double> times{now + horizon}, positions{now + horizon}, final{rest ? 0.0 : kNaN};
      EXPECT_EQ(buffer.set_commands(now, times, positions, final), Status::kOk);
      for (double t = now; t < now + 0.005 && now > 1.0; t += 0.0005)
      {
        low = std::min(low, query(buffer, t).v[0]);
        high = std::max(high, query(buffer, t).v[0]);
      }
    }
    return std::pair{low, high};
  };
  for (bool rest : {false, true})
  {
    for (double horizon : {0.02, 0.1})
    {
      const auto [low, high] = speed_range(rest, horizon);
      EXPECT_NEAR(low, 1.0, rest ? 0.05 : 0.01) << horizon << " " << rest;
      EXPECT_NEAR(high, 1.0, rest ? 0.05 : 0.01) << horizon << " " << rest;
    }
  }
}

TEST(TrajectoryBuffer, NewerCommandReplacesLaterTargets)
{
  auto buffer = make(1);
  ASSERT_EQ(buffer.set_commands(0.0, std::vector<double>{1.0, 3.0}, std::vector<double>{1.0, 3.0}), Status::kOk);
  EXPECT_EQ(buffer.num_targets(0), 2u);
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 3.0);
  ASSERT_EQ(buffer.set_commands(0.5, std::vector<double>{1.5}, std::vector<double>{-1.0}), Status::kOk);
  EXPECT_EQ(buffer.num_targets(0), 2u);  // 1.0, 1.5
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 1.5);
  EXPECT_EQ(query(buffer, 2.0).p[0], -1.0);
  // A command at 0.8 replaces both.
  ASSERT_EQ(buffer.set_commands(0.6, std::vector<double>{0.8, 0.9}, std::vector<double>{4.0, 5.0}), Status::kOk);
  EXPECT_EQ(buffer.num_targets(0), 2u);  // 0.8, 0.9
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 0.9);
  EXPECT_EQ(query(buffer, 1.0).p[0], 5.0);
}

// The targets, unfitted: linear between them (skipping the free knot), held after the last.
TEST(TrajectoryBuffer, QueryTargetInterpolatesTheTargets)
{
  const auto target = [](const TrajectoryBuffer & buffer, double t)
  {
    std::vector<double> p(buffer.num_joints());
    buffer.query_target(t, p);
    return p;
  };
  auto buffer = make(1);
  ASSERT_EQ(buffer.set_commands(0.0, std::vector<double>{1.0, 3.0}, std::vector<double>{1.0, 3.0}), Status::kOk);
  EXPECT_DOUBLE_EQ(target(buffer, 0.0)[0], 0.0);
  EXPECT_DOUBLE_EQ(target(buffer, 0.25)[0], 0.25);  // across the free knot at 0.5
  EXPECT_DOUBLE_EQ(target(buffer, 1.0)[0], 1.0);
  EXPECT_DOUBLE_EQ(target(buffer, 2.0)[0], 2.0);
  EXPECT_DOUBLE_EQ(target(buffer, 5.0)[0], 3.0);

  // A refit at 2.0 keeps the targets around it: still the line between them, not the fit.
  ASSERT_EQ(buffer.set_commands(2.0, std::vector<double>{4.0}, std::vector<double>{5.0}), Status::kOk);
  EXPECT_DOUBLE_EQ(target(buffer, 2.0)[0], 2.0);
  EXPECT_DOUBLE_EQ(target(buffer, 3.5)[0], 4.0);
  EXPECT_DOUBLE_EQ(target(buffer, 10.0)[0], 5.0);
  // A command replaces the timeline from its first time on.
  ASSERT_EQ(buffer.set_commands(2.5, std::vector<double>{3.0}, std::vector<double>{-1.0}), Status::kOk);
  EXPECT_DOUBLE_EQ(target(buffer, 3.0)[0], -1.0);
  EXPECT_DOUBLE_EQ(target(buffer, 10.0)[0], -1.0);

  // The latency delays the targets.
  auto delayed = make(1, {}, 256, 0.1);
  ASSERT_EQ(delayed.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{2.0}), Status::kOk);
  EXPECT_DOUBLE_EQ(target(delayed, 0.55)[0], 1.0);

  // A joint not commanded (another joint is) keeps its targets.
  auto two = make(2);
  ASSERT_EQ(two.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{1.0, kNaN}), Status::kOk);
  ASSERT_EQ(two.set_commands(0.7, std::vector<double>{2.0}, std::vector<double>{kNaN, 1.0}), Status::kOk);
  const double pruned = target(two, 0.85)[0];
  EXPECT_TRUE(std::isfinite(pruned));
  EXPECT_GT(pruned, 0.0);
  EXPECT_LT(pruned, 1.0);
  EXPECT_DOUBLE_EQ(target(two, 1.0)[0], 1.0);
}

// A square wave streamed like teleop (a target 20 ms ahead every 5 ms, refit each time, queried at
// 1 kHz): the fit may overshoot the steps, the targets never leave [low, high].
TEST(TrajectoryBuffer, QueryTargetDoesNotOvershootAStreamedSquareWave)
{
  auto buffer = make(1, Weights{1.0, 0.0, 1e-4}, 10000, 0.05);
  const double low = 0.0, high = 0.3;
  const auto square = [&](double t) { return std::fmod(t * 0.5, 1.0) < 0.5 ? low : high; };
  double fit_max = -1e9;
  std::vector<double> p(1);
  for (int tick = 0; tick < 3000; ++tick)
  {
    const double now = 0.001 * tick;
    if (tick % 5 == 0)
    {
      ASSERT_NE(
        buffer.set_commands(now, std::vector<double>{now + 0.02}, std::vector<double>{square(now + 0.02)}),
        Status::kInvalid);
    }
    buffer.query_target(now, p);
    ASSERT_GE(p[0], low - 1e-9) << "at " << now;
    ASSERT_LE(p[0], high + 1e-9) << "at " << now;
    fit_max = std::max(fit_max, query(buffer, now).p[0]);
  }
  EXPECT_GT(fit_max, high);  // the fit does overshoot: the test exercises the case
}

TEST(TrajectoryBuffer, PreemptionIsContinuous)
{
  auto buffer = make(1, {1.0, 0.0, 1e-4});
  std::vector<double> times, positions;
  for (int k = 1; k <= 50; ++k)
  {
    times.push_back(0.02 * k);
    positions.push_back(std::sin(0.02 * k * 3));
  }
  ASSERT_EQ(buffer.set_commands(0.0, times, positions), Status::kOk);
  const double now = 0.333;
  const auto before = query(buffer, now);
  ASSERT_EQ(buffer.set_commands(now, std::vector<double>{0.4, 0.6}, std::vector<double>{-1.0, -2.0}), Status::kOk);
  const auto after = query(buffer, now);
  EXPECT_NEAR(after.p[0], before.p[0], 1e-12);
  EXPECT_NEAR(after.v[0], before.v[0], 1e-9);
  EXPECT_NEAR(after.a[0], before.a[0], 1e-6);
  EXPECT_NE(before.v[0], 0.0);
}

TEST(TrajectoryBuffer, PartialCommandsLeaveOtherJointsAlone)
{
  auto buffer = make(2);
  ASSERT_EQ(
    buffer.set_commands(0.0, std::vector<double>{1.0, 2.0}, std::vector<double>{1.0, 1.0, 2.0, 2.0}),
    Status::kOk);
  const auto before = query(buffer, 1.3);
  ASSERT_EQ(buffer.set_commands(0.5, std::vector<double>{1.0}, std::vector<double>{kNaN, -1.0}), Status::kOk);
  const auto after = query(buffer, 1.3);
  EXPECT_EQ(after.p[0], before.p[0]);
  EXPECT_EQ(after.v[0], before.v[0]);
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 2.0);
  EXPECT_DOUBLE_EQ(buffer.end_time(1), 1.0);
  EXPECT_EQ(query(buffer, 1.3).p[1], -1.0);
  // A joint left out of some points of a command only gets the points that name it.
  ASSERT_EQ(
    buffer.set_commands(2.5, std::vector<double>{3.0, 3.5}, std::vector<double>{5.0, kNaN, kNaN, 7.0}),
    Status::kOk);
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 3.0);
  EXPECT_DOUBLE_EQ(buffer.end_time(1), 3.5);
}

TEST(TrajectoryBuffer, TrajectoryIsTwiceContinuouslyDifferentiable)
{
  auto buffer = make(1, {1.0, 0.1, 1e-4});
  std::vector<double> times, positions;
  for (int k = 1; k <= 20; ++k)
  {
    times.push_back(0.05 * k);
    positions.push_back((k % 3) * 0.1);
  }
  ASSERT_EQ(buffer.set_commands(0.0, times, positions, std::vector<double>{0.0}), Status::kOk);
  for (double t : times)
  {
    const auto l = query(buffer, t - 1e-9), r = query(buffer, t + 1e-9);
    EXPECT_NEAR(l.p[0], r.p[0], 1e-6) << t;
    EXPECT_NEAR(l.v[0], r.v[0], 1e-4) << t;
    EXPECT_NEAR(l.a[0], r.a[0], 1e-2 * std::max(1.0, std::abs(l.a[0]))) << t;
  }
}

TEST(TrajectoryBuffer, WeightsTradeTrackingForSmoothness)
{
  // A sine sampled every 20 ms; measured in the middle, away from the rest-to-rest ends.
  const auto run = [](Weights w, double & max_error, double & max_acceleration)
  {
    auto buffer = make(1, w);
    std::vector<double> times, positions;
    for (int k = 1; k <= 100; ++k)
    {
      times.push_back(0.02 * k);
      positions.push_back(0.5 * (1 - std::cos(2 * M_PI * 0.02 * k)));
    }
    EXPECT_EQ(buffer.set_commands(0.0, times, positions), Status::kOk);
    max_error = max_acceleration = 0;
    for (std::size_t k = 25; k < 75; ++k)
    {
      const auto s = query(buffer, times[k]);
      max_error = std::max(max_error, std::abs(s.p[0] - positions[k]));
      max_acceleration = std::max(max_acceleration, std::abs(s.a[0]));
    }
  };
  double tight_error, tight_acc, smooth_error, smooth_acc;
  run({1.0, 0.0, 1e-9}, tight_error, tight_acc);
  run({1.0, 0.0, 1e-2}, smooth_error, smooth_acc);
  EXPECT_LT(tight_error, 1e-4);
  EXPECT_NEAR(tight_acc, 0.5 * 4 * M_PI * M_PI, 0.5);  // the sine's own peak acceleration
  EXPECT_GT(smooth_error, tight_error);
  EXPECT_LT(smooth_acc, tight_acc);
}

TEST(TrajectoryBuffer, PastTargetsAndCloseTargets)
{
  auto buffer = make(1);
  // All due already: the last one is kept, min_interval after now.
  ASSERT_EQ(buffer.set_commands(1.0, std::vector<double>{0.5, 0.9}, std::vector<double>{3.0, 4.0}), Status::kOk);
  EXPECT_EQ(buffer.num_targets(0), 1u);
  EXPECT_NEAR(buffer.end_time(0), 1.001, 1e-12);
  EXPECT_EQ(query(buffer, 1.01).p[0], 4.0);
  // Some due: those are dropped.
  ASSERT_EQ(buffer.set_commands(2.0, std::vector<double>{2.0, 2.5}, std::vector<double>{0.0, 1.0}), Status::kOk);
  EXPECT_EQ(buffer.num_targets(0), 1u);
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 2.5);
}

TEST(TrajectoryBuffer, TruncatesAtCapacity)
{
  auto buffer = make(1, {}, 4);
  std::vector<double> times, positions;
  for (int k = 1; k <= 6; ++k)
  {
    times.push_back(k);
    positions.push_back(k);
  }
  EXPECT_EQ(buffer.set_commands(0.0, times, positions), Status::kTruncated);
  EXPECT_EQ(buffer.num_targets(0), 4u);
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 4.0);
}

TEST(TrajectoryBuffer, RejectsInvalidCommands)
{
  auto buffer = make(2);
  const double inf = std::numeric_limits<double>::infinity();
  EXPECT_EQ(buffer.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{1.0}), Status::kInvalid);
  EXPECT_EQ(
    buffer.set_commands(0.0, std::vector<double>{1.0, 1.0}, std::vector<double>{1, 1, 1, 1}), Status::kInvalid);
  EXPECT_EQ(buffer.set_commands(0.0, std::vector<double>{kNaN}, std::vector<double>{1, 1}), Status::kInvalid);
  EXPECT_EQ(buffer.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{inf, 1}), Status::kInvalid);
  EXPECT_EQ(
    buffer.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{1, 1}, std::vector<double>{0.0}),
    Status::kInvalid);
  EXPECT_EQ(buffer.num_targets(0), 0u);
  EXPECT_THROW(make(1, {0.0, 0.0, 1.0}), std::invalid_argument);
  EXPECT_THROW(make(1, {1.0, 0.0, 0.0}), std::invalid_argument);
}

TEST(TrajectoryBuffer, LatencyDelaysCommandsSoStreamsOverlap)
{
  auto buffer = make(1, {}, 256, 0.1);
  ASSERT_EQ(buffer.set_commands(0.0, std::vector<double>{0.5}, std::vector<double>{1.0}), Status::kOk);
  EXPECT_DOUBLE_EQ(buffer.end_time(0), 0.6);
  EXPECT_LT(query(buffer, 0.5).p[0], 1.0);
  EXPECT_EQ(query(buffer, 0.6).p[0], 1.0);
  // A stream of one target 20 ms ahead every 5 ms keeps ~ (latency + 20 ms) / 5 ms of them.
  for (int cycle = 0; cycle < 100; ++cycle)
  {
    const double now = 1.0 + 0.005 * cycle;
    ASSERT_EQ(buffer.set_commands(now, std::vector<double>{now + 0.02}, std::vector<double>{now}), Status::kOk);
  }
  EXPECT_GE(buffer.num_targets(0), 20u);
  EXPECT_LE(buffer.num_targets(0), 25u);
  EXPECT_THROW(buffer.set_latency(-1.0), std::invalid_argument);
}

TEST(TrajectoryBuffer, PastKnotsAreDroppedFromEveryJoint)
{
  auto buffer = make(2);
  std::vector<double> times, positions;
  for (int k = 1; k <= 10; ++k)
  {
    times.push_back(0.1 * k);
    positions.insert(positions.end(), {kNaN, 0.1 * k});
  }
  ASSERT_EQ(buffer.set_commands(0.0, times, positions), Status::kOk);
  const std::size_t knots = buffer.num_knots(1);
  const auto before = query(buffer, 0.55);
  // Commands for joint 0 alone drop joint 1's passed knots, keeping its trajectory.
  ASSERT_EQ(buffer.set_commands(0.55, std::vector<double>{1.0}, std::vector<double>{1.0, kNaN}), Status::kOk);
  EXPECT_LT(buffer.num_knots(1), knots);
  EXPECT_EQ(buffer.num_targets(1), 5u);
  const auto after = query(buffer, 0.55);
  EXPECT_EQ(after.p[1], before.p[1]);
  EXPECT_EQ(after.v[1], before.v[1]);
  ASSERT_EQ(buffer.set_commands(2.0, std::vector<double>{3.0}, std::vector<double>{1.0, kNaN}), Status::kOk);
  EXPECT_EQ(buffer.num_knots(1), 1u);  // all passed: holding
  EXPECT_NEAR(query(buffer, 5.0).p[1], 1.0, 1e-12);
}

// The largest position over [t0, t1], sampled every 0.1 ms.
double peak(const TrajectoryBuffer & buffer, double t0, double t1)
{
  double high = -1e9;
  for (double t = t0; t <= t1; t += 1e-4) high = std::max(high, query(buffer, t).p[0]);
  return high;
}

TEST(TrajectoryBuffer, TargetsAreClampedIntoTheLimits)
{
  auto buffer = make(1, {}, 256, 0.0, {-1.0}, {1.0});
  ASSERT_EQ(buffer.set_commands(0.0, std::vector<double>{1.0}, std::vector<double>{2.0}), Status::kOk);
  EXPECT_EQ(query(buffer, 2.0).p[0], 1.0);
  EXPECT_LE(peak(buffer, 0.0, 1.0), 1.0 + 1e-9);
}

// Moving at 3 rad/s 5 mrad from the limit, then told to stop at the limit 20 ms later:
// unconstrained the trajectory overshoots it; the QP brakes in time (900 rad/s^2 would do).
TEST(TrajectoryBuffer, LimitsHoldWhenArrivingFast)
{
  const auto run = [](bool limited)
  {
    auto buffer = limited ? make(1, {}, 256, 0.0, {-10.0}, {0.905}) : make(1);
    buffer.reset(0.0, std::vector<double>{0.0});
    std::vector<double> times, positions;
    for (int k = 1; k <= 20; ++k)
    {
      times.push_back(0.02 * k);
      positions.push_back(0.06 * k);  // a ramp at 3 rad/s, within the limit until 0.3 s
    }
    EXPECT_NE(buffer.set_commands(0.0, times, positions), Status::kInvalid);
    const std::size_t iterations = buffer.qp_iterations();
    // At 0.3 s (p ~ 0.9, 3 rad/s), a stop at the limit 0.02 s later.
    EXPECT_NE(
      buffer.set_commands(0.3, std::vector<double>{0.32}, std::vector<double>{0.905}, std::vector<double>{0.0}),
      Status::kInvalid);
    return std::pair{peak(buffer, 0.3, 1.0), iterations + buffer.qp_iterations()};
  };
  const auto [free_peak, free_iterations] = run(false);
  const auto [limited_peak, limited_iterations] = run(true);
  EXPECT_GT(free_peak, 0.91);
  EXPECT_EQ(free_iterations, 0u);
  EXPECT_LE(limited_peak, 0.905 + 1e-3);
  EXPECT_GT(limited_iterations, 0u);
}

// Moving out at 0.75 rad/s 0.01 rad inside the lower limit, then a long move to the other end
// (as when IK switches branch): the 2 s segments would bulge past the limit between the fixed
// sample points; the dense check adds rows there.
TEST(TrajectoryBuffer, LongSegmentsStayWithinLimits)
{
  auto buffer = make(1, {}, 256, 0.0, {-3.14}, {3.14});
  buffer.reset(0.0, std::vector<double>{-2.0});
  std::vector<double> times, positions;
  for (int k = 1; k <= 76; ++k)
  {
    times.push_back(0.02 * k);
    positions.push_back(-2.0 - 0.015 * k);  // -0.75 rad/s, to -3.14
  }
  ASSERT_EQ(buffer.set_commands(0.0, times, positions), Status::kOk);
  const double now = 1.5;  // at about -3.125, still moving out
  ASSERT_LT(query(buffer, now).v[0], -0.3);
  const auto status = buffer.set_commands(now, std::vector<double>{now + 4.0}, std::vector<double>{3.1}, std::vector<double>{0.0});
  EXPECT_EQ(status, Status::kOk);
  double low = 1e9;
  for (double t = now; t <= now + 4.0; t += 1e-4) low = std::min(low, query(buffer, t).p[0]);
  EXPECT_GE(low, -3.14 - 1e-4);
  EXPECT_NEAR(query(buffer, now + 4.0).p[0], 3.1, 1e-9);
}

// A dense trajectory (a sine past the upper limit): within the limits, and following the targets
// where they are inside them.
TEST(TrajectoryBuffer, DenseTrajectoryStaysWithinLimits)
{
  auto buffer = make(1, {1.0, 0.0, 1e-6}, 256, 0.0, {-0.8}, {0.8});
  std::vector<double> times, positions;
  for (int k = 1; k <= 100; ++k)
  {
    times.push_back(0.02 * k);
    positions.push_back(std::sin(M_PI * 0.02 * k));
  }
  ASSERT_EQ(buffer.set_commands(0.0, times, positions, std::vector<double>{0.0}), Status::kOk);
  EXPECT_GT(buffer.qp_iterations(), 0u);
  EXPECT_LE(peak(buffer, 0.0, 2.0), 0.8 + 1e-3);
  for (std::size_t k = 0; k < times.size(); ++k)
  {
    if (std::abs(positions[k]) < 0.7) EXPECT_NEAR(query(buffer, times[k]).p[0], positions[k], 0.02) << times[k];
  }
}

// Every refit starts from the previous fit's state: across random preemptions (partial commands,
// latency, active limits, free and rest ends, targets due already) position, velocity and
// acceleration don't jump at `now`.
TEST(TrajectoryBuffer, RefitsAreContinuous)
{
  constexpr std::size_t kJoints = 3;
  auto buffer = make(
    kJoints, {1.0, 1e-3, 1e-5}, 256, 0.03, std::vector<double>(kJoints, -1.0), std::vector<double>(kJoints, 1.0));
  std::srand(7);
  const auto uniform = [](double lo, double hi) { return lo + (hi - lo) * std::rand() / RAND_MAX; };
  double now = 0.0, worst_p = 0, worst_v = 0, worst_a = 0;
  for (int cycle = 0; cycle < 2000; ++cycle)
  {
    now += uniform(0.001, 0.03);
    const std::size_t points = 1 + static_cast<std::size_t>(std::rand() % 6);
    std::vector<double> times, positions, final(kJoints);
    double t = now + uniform(-0.05, 0.05);
    for (std::size_t k = 0; k < points; ++k)
    {
      t += uniform(0.005, 0.1);
      times.push_back(t);
      for (std::size_t j = 0; j < kJoints; ++j) positions.push_back(std::rand() % 4 == 0 ? kNaN : uniform(-1.3, 1.3));
    }
    for (auto & v : final) v = std::rand() % 2 ? kNaN : 0.0;
    const auto before = query(buffer, now);
    ASSERT_NE(buffer.set_commands(now, times, positions, final), Status::kInvalid);
    const auto after = query(buffer, now);
    for (std::size_t j = 0; j < kJoints; ++j)
    {
      worst_p = std::max(worst_p, std::abs(after.p[j] - before.p[j]));
      worst_v = std::max(worst_v, std::abs(after.v[j] - before.v[j]));
      worst_a = std::max(worst_a, std::abs(after.a[j] - before.a[j]));
    }
  }
  EXPECT_LT(worst_p, 1e-12);
  EXPECT_LT(worst_v, 1e-9);
  EXPECT_LT(worst_a, 1e-6);
}

TEST(TrajectoryBuffer, DoesNotAllocateAfterConstruction)
{
  constexpr std::size_t kJoints = 7, kPoints = 50;
  // Limits the sines pass, so the QP runs.
  auto buffer = make(
    kJoints, {1.0, 0.01, 1e-5}, 256, 0.05, std::vector<double>(kJoints, -0.9), std::vector<double>(kJoints, 0.9));
  std::vector<double> times(kPoints), positions(kPoints * kJoints), p(kJoints), v(kJoints), a(kJoints);
  for (std::size_t k = 0; k < kPoints; ++k)
  {
    times[k] = 0.02 * (k + 1);
    for (std::size_t j = 0; j < kJoints; ++j) positions[k * kJoints + j] = std::sin(0.1 * k + j);
  }
  const long before = g_allocations.load();
  double now = 0.0;
  for (int cycle = 0; cycle < 100; ++cycle, now += 0.002)
  {
    for (auto & t : times) t += 0.002;
    ASSERT_EQ(buffer.set_commands(now, times, positions), Status::kOk);
    buffer.query_command(now, p, v, a);
  }
  EXPECT_EQ(g_allocations.load(), before);
  EXPECT_GT(buffer.qp_iterations(), 0u);
}

TEST(TrajectoryBuffer, Timing)
{
  constexpr std::size_t kJoints = 7, kPoints = 50;
  auto buffer = make(kJoints, {1.0, 0.0, 1e-6});
  std::vector<double> times(kPoints), positions(kPoints * kJoints), p(kJoints), v(kJoints), a(kJoints);
  for (std::size_t k = 0; k < kPoints; ++k)
  {
    times[k] = 0.02 * (k + 1);
    for (std::size_t j = 0; j < kJoints; ++j) positions[k * kJoints + j] = std::sin(0.1 * k + j);
  }
  constexpr int kCycles = 2000;
  const auto t0 = std::chrono::steady_clock::now();
  for (int cycle = 0; cycle < kCycles; ++cycle)
  {
    (void)buffer.set_commands(0.0, times, positions);
  }
  const auto t1 = std::chrono::steady_clock::now();
  for (int cycle = 0; cycle < kCycles; ++cycle) buffer.query_command(0.001 * (cycle % 1000), p, v, a);
  const auto t2 = std::chrono::steady_clock::now();
  const auto us = [](auto d) { return std::chrono::duration<double, std::micro>(d).count() / kCycles; };
  auto limited = make(
    kJoints, {1.0, 0.0, 1e-6}, 256, 0.0, std::vector<double>(kJoints, -0.9), std::vector<double>(kJoints, 0.9));
  const auto t3 = std::chrono::steady_clock::now();
  for (int cycle = 0; cycle < kCycles / 10; ++cycle) (void)limited.set_commands(0.0, times, positions);
  const auto t4 = std::chrono::steady_clock::now();
  std::printf(
    "set_commands (%zu joints x %zu points): %.2f us, with active limits %.2f us (%zu QP iterations); "
    "query_command: %.3f us\n",
    kJoints, kPoints, us(t1 - t0), 10 * us(t4 - t3), limited.qp_iterations(), us(t2 - t1));
}

}  // namespace
