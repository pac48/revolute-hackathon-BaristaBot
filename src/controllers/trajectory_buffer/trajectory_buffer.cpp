#include "trajectory_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace trajectory_buffer
{

namespace
{
constexpr std::size_t kBand = 6;  // half-bandwidth + 1 of the system: knots couple to their neighbors' 3 values
constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr std::size_t kDenseChecks = 32;  // grid per segment on which to find its extrema
constexpr std::size_t kLimitRounds = 8;   // of adding rows where the solution is outside
constexpr double kLimitTolerance = 1e-6;  // rad (or m) outside a limit that counts as within

// Coefficients (in s = (t - t0) / h, s in [0, 1]) of the quintic with scaled end values
// x = (p0, v0 h, a0 h^2, p1, v1 h, a1 h^2): c = kHermite x.
constexpr double kHermite[6][6] = {
  {1, 0, 0, 0, 0, 0},
  {0, 1, 0, 0, 0, 0},
  {0, 0, 0.5, 0, 0, 0},
  {-10, -6, -1.5, 10, -4, 0.5},
  {15, 8, 1.5, -15, 7, -1},
  {-6, -3, -0.5, 6, -3, 0.5},
};

void check(const Weights & w)
{
  if (!(w.position > 0.0) || !(w.velocity >= 0.0) || !(w.acceleration >= 0.0) ||
      !(w.velocity > 0.0 || w.acceleration > 0.0) || !std::isfinite(w.position) ||
      !std::isfinite(w.velocity) || !std::isfinite(w.acceleration))
  {
    throw std::invalid_argument(
      "trajectory_buffer: weights need position > 0 and velocity or acceleration > 0 (all finite)");
  }
}

void check_latency(double latency)
{
  if (!(latency >= 0.0) || !std::isfinite(latency))
    throw std::invalid_argument("trajectory_buffer: latency must be finite and >= 0");
}

// c^T Q c of the quintic's derivative of the given order over s in [0, 1], as a matrix on x.
void segment_cost(int order, double out[6][6])
{
  double q[6][6] = {};
  for (int i = order; i < 6; ++i)
  {
    for (int j = order; j < 6; ++j)
    {
      double fi = 1, fj = 1;
      for (int k = 0; k < order; ++k)
      {
        fi *= i - k;
        fj *= j - k;
      }
      q[i][j] = fi * fj / (i + j - 2 * order + 1);
    }
  }
  for (int a = 0; a < 6; ++a)
  {
    for (int b = 0; b < 6; ++b)
    {
      double sum = 0;
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) sum += kHermite[i][a] * q[i][j] * kHermite[j][b];
      out[a][b] = sum;
    }
  }
}

std::size_t band_start(std::size_t i) { return i >= kBand - 1 ? i - (kBand - 1) : 0; }

// In place: the lower band of a symmetric positive definite matrix (band[i * kBand + d] = H(i, i - d))
// becomes its Cholesky factor L. False if not positive definite.
bool cholesky(double * band, std::size_t n)
{
  for (std::size_t i = 0; i < n; ++i)
  {
    const std::size_t lo = band_start(i);
    for (std::size_t j = lo; j <= i; ++j)
    {
      double sum = band[i * kBand + (i - j)];
      for (std::size_t m = lo; m < j; ++m) sum -= band[i * kBand + (i - m)] * band[j * kBand + (j - m)];
      if (j == i)
      {
        if (!(sum > 0.0)) return false;
        band[i * kBand] = std::sqrt(sum);
      }
      else
      {
        band[i * kBand + (i - j)] = sum / band[j * kBand];
      }
    }
  }
  return true;
}

// In place: x = (L L^T)^-1 x.
void cholesky_solve(const double * band, std::size_t n, double * x)
{
  for (std::size_t i = 0; i < n; ++i)
  {
    double sum = x[i];
    for (std::size_t m = band_start(i); m < i; ++m) sum -= band[i * kBand + (i - m)] * x[m];
    x[i] = sum / band[i * kBand];
  }
  for (std::size_t i = n; i-- > 0;)
  {
    double sum = x[i];
    for (std::size_t m = i + 1; m < std::min(n, i + kBand); ++m) sum -= band[m * kBand + (m - i)] * x[m];
    x[i] = sum / band[i * kBand];
  }
}

// y = H x for the symmetric band H.
void band_multiply(const double * band, std::size_t n, const double * x, double * y)
{
  for (std::size_t i = 0; i < n; ++i) y[i] = band[i * kBand] * x[i];
  for (std::size_t i = 0; i < n; ++i)
  {
    for (std::size_t j = band_start(i); j < i; ++j)
    {
      const double h = band[i * kBand + (i - j)];
      y[i] += h * x[j];
      y[j] += h * x[i];
    }
  }
}
}  // namespace

TrajectoryBuffer::TrajectoryBuffer(const Options & options) : options_(options)
{
  const std::size_t dof = options_.num_joints;
  if (dof == 0) throw std::invalid_argument("trajectory_buffer: num_joints must be > 0");
  if (options_.capacity == 0) throw std::invalid_argument("trajectory_buffer: capacity must be > 0");
  if (!(options_.min_interval > 0.0) || !std::isfinite(options_.min_interval))
    throw std::invalid_argument("trajectory_buffer: min_interval must be > 0");
  if (options_.limit_samples == 0) throw std::invalid_argument("trajectory_buffer: limit_samples must be > 0");
  check(options_.weights);
  check_latency(options_.latency);
  if (options_.lower.empty()) options_.lower.assign(dof, -kInf);
  if (options_.upper.empty()) options_.upper.assign(dof, kInf);
  if (options_.lower.size() != dof || options_.upper.size() != dof)
    throw std::invalid_argument("trajectory_buffer: lower and upper need a limit per joint (or none)");
  for (std::size_t j = 0; j < dof; ++j)
  {
    if (std::isnan(options_.lower[j]) || std::isnan(options_.upper[j]) || options_.lower[j] > options_.upper[j])
      throw std::invalid_argument("trajectory_buffer: each joint needs lower <= upper");
  }

  // Knots: the start, a free knot and capacity targets.
  const std::size_t knots = options_.capacity + 2;
  joints_.resize(dof);
  for (auto & joint : joints_)
  {
    for (auto * v : {&joint.time, &joint.target, &joint.position, &joint.velocity, &joint.acceleration})
      v->assign(knots, 0.0);
    joint.size = 1;
    // Timeline: the last passed target and capacity ahead.
    joint.timeline_time.assign(options_.capacity + 1, 0.0);
    joint.timeline_target.assign(options_.capacity + 1, 0.0);
    joint.timeline_size = 1;
  }
  const std::size_t unknowns = 3 * (knots - 2) + 1;
  // Sample rows, then per round up to 4 extrema of each segment's quintic.
  const std::size_t rows = (2 * (options_.limit_samples + 1) + 4 * kLimitRounds) * (knots - 1);
  for (auto * v : {&hessian_, &band_}) v->assign(unknowns * kBand, 0.0);
  for (auto * v : {&rhs_, &x_, &dx_, &rd_}) v->assign(unknowns, 0.0);
  for (auto * v : {&row_bound_, &slack_, &dual_, &dslack_, &ddual_, &rp_, &rc_}) v->assign(rows, 0.0);
  row_coef_.assign(6 * rows, 0.0);
  row_index_.assign(6 * rows, -1);
  segment_cost(1, velocity_cost_);
  segment_cost(2, acceleration_cost_);
}

void TrajectoryBuffer::set_weights(const Weights & weights)
{
  check(weights);
  options_.weights = weights;
}

void TrajectoryBuffer::set_latency(double latency)
{
  check_latency(latency);
  options_.latency = latency;
}

void TrajectoryBuffer::reset(double time, std::span<const double> positions)
{
  if (positions.size() != options_.num_joints)
    throw std::invalid_argument("trajectory_buffer: reset needs a position per joint");
  for (std::size_t j = 0; j < joints_.size(); ++j)
  {
    auto & joint = joints_[j];
    joint.size = 1;
    joint.time[0] = time;
    joint.target[0] = joint.position[0] = positions[j];
    joint.velocity[0] = joint.acceleration[0] = 0.0;
    joint.final_velocity = 0.0;
    joint.timeline_size = 1;
    joint.timeline_time[0] = time;
    joint.timeline_target[0] = positions[j];
  }
}

double TrajectoryBuffer::end_time(std::size_t joint) const
{
  const auto & j = joints_[joint];
  return j.time[j.size - 1];
}

double TrajectoryBuffer::end_time() const
{
  double end = end_time(0);
  for (std::size_t j = 1; j < joints_.size(); ++j) end = std::max(end, end_time(j));
  return end;
}

std::size_t TrajectoryBuffer::num_targets(std::size_t joint) const
{
  const auto & j = joints_[joint];
  std::size_t count = 0;
  for (std::size_t i = 1; i < j.size; ++i) count += std::isnan(j.target[i]) ? 0 : 1;
  return count;
}

TrajectoryBuffer::State TrajectoryBuffer::evaluate(const Joint & joint, double time) const
{
  const std::size_t last = joint.size - 1;
  if (time >= joint.time[last]) return {joint.position[last], 0.0, 0.0};  // holding
  if (time <= joint.time[0]) return {joint.position[0], joint.velocity[0], joint.acceleration[0]};
  const std::size_t k =
    static_cast<std::size_t>(std::upper_bound(joint.time.begin(), joint.time.begin() + last, time) - joint.time.begin()) - 1;
  const double h = joint.time[k + 1] - joint.time[k];
  const double x[6] = {
    joint.position[k],     joint.velocity[k] * h,     joint.acceleration[k] * h * h,
    joint.position[k + 1], joint.velocity[k + 1] * h, joint.acceleration[k + 1] * h * h,
  };
  double c[6];
  for (int i = 0; i < 6; ++i)
  {
    c[i] = 0;
    for (int l = 0; l < 6; ++l) c[i] += kHermite[i][l] * x[l];
  }
  const double s = (time - joint.time[k]) / h;
  double p = c[5], v = 5 * c[5], a = 20 * c[5];
  for (int i = 4; i >= 0; --i) p = p * s + c[i];
  for (int i = 4; i >= 1; --i) v = v * s + i * c[i];
  for (int i = 4; i >= 2; --i) a = a * s + i * (i - 1) * c[i];
  return {p, v / h, a / (h * h)};
}

void TrajectoryBuffer::query_command(
  double time, std::span<double> positions, std::span<double> velocities,
  std::span<double> accelerations) const
{
  for (std::size_t j = 0; j < joints_.size(); ++j)
  {
    const State s = evaluate(joints_[j], time);
    if (j < positions.size()) positions[j] = s.position;
    if (j < velocities.size()) velocities[j] = s.velocity;
    if (j < accelerations.size()) accelerations[j] = s.acceleration;
  }
}

void TrajectoryBuffer::query_target(double time, std::span<double> positions) const
{
  for (std::size_t j = 0; j < joints_.size() && j < positions.size(); ++j)
  {
    const auto & joint = joints_[j];
    const auto begin = joint.timeline_time.begin();
    const std::size_t n = joint.timeline_size;
    const auto after = static_cast<std::size_t>(std::upper_bound(begin, begin + n, time) - begin);
    if (after == 0 || after == n)
    {
      positions[j] = joint.timeline_target[after == 0 ? 0 : n - 1];
      continue;
    }
    const std::size_t before = after - 1;
    const double s = (time - joint.timeline_time[before]) / (joint.timeline_time[after] - joint.timeline_time[before]);
    positions[j] = joint.timeline_target[before] + s * (joint.timeline_target[after] - joint.timeline_target[before]);
  }
}

void TrajectoryBuffer::prune_timeline(Joint & joint, double now)
{
  std::size_t passed = 0;  // the targets at or before now
  while (passed < joint.timeline_size && joint.timeline_time[passed] <= now) ++passed;
  if (passed < 2) return;
  const std::size_t drop = passed - 1;
  for (std::size_t i = drop; i < joint.timeline_size; ++i)
  {
    joint.timeline_time[i - drop] = joint.timeline_time[i];
    joint.timeline_target[i - drop] = joint.timeline_target[i];
  }
  joint.timeline_size -= drop;
}

bool TrajectoryBuffer::append_timeline(Joint & joint, double now, double t, double y)
{
  std::size_t & n = joint.timeline_size;
  if (n > 0 && t <= now && joint.timeline_time[n - 1] <= now) --n;  // only the last passed one matters
  if (n == joint.timeline_time.size()) return false;
  joint.timeline_time[n] = t;
  joint.timeline_target[n] = y;
  ++n;
  return true;
}

void TrajectoryBuffer::prune(Joint & joint, double now) const
{
  // Keep the segment `now` is in and the ones after it.
  std::size_t ahead = 0;
  while (ahead < joint.size && joint.time[ahead] <= now) ++ahead;
  if (ahead == joint.size)
  {
    // All passed: hold the last knot.
    const std::size_t last = joint.size - 1;
    joint.time[0] = joint.time[last];
    joint.target[0] = joint.target[last];
    joint.position[0] = joint.position[last];
    joint.velocity[0] = joint.acceleration[0] = 0.0;
    joint.size = 1;
    return;
  }
  if (ahead < 2) return;
  const std::size_t drop = ahead - 1;
  for (std::size_t i = drop; i < joint.size; ++i)
  {
    joint.time[i - drop] = joint.time[i];
    joint.target[i - drop] = joint.target[i];
    joint.position[i - drop] = joint.position[i];
    joint.velocity[i - drop] = joint.velocity[i];
    joint.acceleration[i - drop] = joint.acceleration[i];
  }
  joint.size -= drop;
}

Status TrajectoryBuffer::set_commands(
  double now, std::span<const double> times, std::span<const double> positions,
  std::span<const double> final_velocities)
{
  const std::size_t dof = options_.num_joints;
  if (positions.size() != times.size() * dof || !std::isfinite(now)) return Status::kInvalid;
  if (!final_velocities.empty() && final_velocities.size() != dof) return Status::kInvalid;
  for (const double v : final_velocities)
  {
    if (std::isinf(v)) return Status::kInvalid;
  }
  for (std::size_t k = 0; k < times.size(); ++k)
  {
    if (!std::isfinite(times[k]) || (k > 0 && !(times[k] > times[k - 1]))) return Status::kInvalid;
  }
  for (const double p : positions)
  {
    if (std::isinf(p)) return Status::kInvalid;
  }

  Status status = Status::kOk;
  qp_iterations_ = 0;
  const std::size_t max_size = options_.capacity + 1;  // the start and the targets
  const double latency = options_.latency;
  for (std::size_t j = 0; j < dof; ++j)
  {
    auto & joint = joints_[j];
    std::size_t first = 0;
    while (first < times.size() && std::isnan(positions[first * dof + j])) ++first;
    if (first == times.size())
    {
      prune(joint, now);  // not commanded: keeps its trajectory
      prune_timeline(joint, now);
      continue;
    }
    const double lower = options_.lower[j], upper = options_.upper[j];
    const auto clamp = [&](double y) { return std::min(std::max(y, lower), upper); };

    // The timeline: the command replaces its targets from the command's first time on.
    {
      prune_timeline(joint, now);
      const double replaced_from = times[first] + latency;
      while (joint.timeline_size > 0 && joint.timeline_time[joint.timeline_size - 1] >= replaced_from)
        --joint.timeline_size;
      for (std::size_t k = first; k < times.size(); ++k)
      {
        const double y = positions[k * dof + j];
        if (!std::isnan(y) && !append_timeline(joint, now, times[k] + latency, clamp(y))) status = Status::kTruncated;
      }
    }

    // Start from where the trajectory is now.
    const State start = evaluate(joint, now);
    // Keep the old targets still ahead (by min_interval: closer ones are as good as reached) and
    // before the command's first one; drop the rest.
    const double first_time = times[first] + latency;
    std::size_t size = 1;
    for (std::size_t i = 1; i < joint.size; ++i)
    {
      if (!std::isnan(joint.target[i]) && joint.time[i] >= now + options_.min_interval && joint.time[i] < first_time)
      {
        joint.time[size] = joint.time[i];
        joint.target[size] = joint.target[i];
        ++size;
      }
    }
    joint.time[0] = now;
    joint.target[0] = joint.position[0] = start.position;
    joint.velocity[0] = start.velocity;
    joint.acceleration[0] = start.acceleration;

    const auto append = [&](double time, double target)
    {
      if (size == max_size)
      {
        status = Status::kTruncated;
        return;
      }
      joint.time[size] = std::max(time, joint.time[size - 1] + options_.min_interval);
      joint.target[size] = clamp(target);
      ++size;
    };
    // Targets already due are dropped (the fit starts now), except a command's last one when it
    // has none ahead: that one is moved to now + min_interval.
    std::size_t late = times.size();
    bool any = false;
    for (std::size_t k = first; k < times.size(); ++k)
    {
      const double y = positions[k * dof + j];
      if (std::isnan(y)) continue;
      if (times[k] + latency <= now)
      {
        late = k;
        continue;
      }
      any = true;
      append(times[k] + latency, y);
    }
    if (!any && late < times.size()) append(now, positions[late * dof + j]);

    // A free knot halfway to the first target, so that even one target leaves the fit a choice.
    if (size > 1 && joint.time[1] - joint.time[0] >= 2 * options_.min_interval)
    {
      for (std::size_t i = size; i > 1; --i)
      {
        joint.time[i] = joint.time[i - 1];
        joint.target[i] = joint.target[i - 1];
      }
      joint.time[1] = 0.5 * (joint.time[0] + joint.time[2]);
      joint.target[1] = std::numeric_limits<double>::quiet_NaN();
      ++size;
    }
    joint.size = size;
    joint.final_velocity =
      final_velocities.empty() ? std::numeric_limits<double>::quiet_NaN() : final_velocities[j];
    if (!fit(joint, lower, upper)) status = Status::kLimitsViolated;
  }
  return status;
}

bool TrajectoryBuffer::fit(Joint & joint, double lower, double upper)
{
  const std::size_t last = joint.size - 1;
  if (last == 0) return true;
  // The last target is reached exactly with zero acceleration, at the final velocity if given.
  const bool free_end = std::isnan(joint.final_velocity);
  joint.position[last] = joint.target[last];
  joint.velocity[last] = free_end ? 0.0 : joint.final_velocity;
  joint.acceleration[last] = 0.0;

  // Unknowns: (p, v, a) of knots 1 .. last - 1 at 3 (k - 1) + c, then the last velocity if free.
  const std::size_t n = 3 * (last - 1) + (free_end ? 1 : 0);
  const auto unknown = [&](std::size_t knot, int c) -> long
  {
    if (knot == 0) return -1;
    if (knot < last) return static_cast<long>(3 * (knot - 1) + static_cast<std::size_t>(c));
    return free_end && c == 1 ? static_cast<long>(n - 1) : -1;
  };
  const auto value = [&](std::size_t knot, int c)
  { return c == 0 ? joint.position[knot] : c == 1 ? joint.velocity[knot] : joint.acceleration[knot]; };

  std::fill(hessian_.begin(), hessian_.begin() + n * kBand, 0.0);
  std::fill(rhs_.begin(), rhs_.begin() + n, 0.0);
  double * hessian = hessian_.data();  // hessian[i * kBand + d] = H(i, i - d)
  double * rhs = rhs_.data();
  const Weights & w = options_.weights;
  for (std::size_t k = 0; k < last; ++k)
  {
    const double h = joint.time[k + 1] - joint.time[k];
    const double scale[6] = {1, h, h * h, 1, h, h * h};
    const double cv = w.velocity / h, ca = w.acceleration / (h * h * h);
    long index[6];
    double fixed[6];
    for (int i = 0; i < 6; ++i)
    {
      const std::size_t knot = k + static_cast<std::size_t>(i / 3);
      index[i] = unknown(knot, i % 3);
      fixed[i] = value(knot, i % 3);
    }
    for (int i = 0; i < 6; ++i)
    {
      if (index[i] < 0) continue;
      for (int l = 0; l < 6; ++l)
      {
        const double s = scale[i] * scale[l] * (cv * velocity_cost_[i][l] + ca * acceleration_cost_[i][l]);
        if (index[l] < 0)
          rhs[index[i]] -= s * fixed[l];
        else if (index[l] <= index[i])
          hessian[index[i] * kBand + (index[i] - index[l])] += s;
      }
    }
  }
  for (std::size_t k = 1; k < last; ++k)
  {
    if (std::isnan(joint.target[k])) continue;  // the free knot
    const std::size_t i = 3 * (k - 1);
    hessian[i * kBand] += w.position;
    rhs[i] += w.position * joint.target[k];
  }

  // The unconstrained minimum.
  std::copy(hessian_.begin(), hessian_.begin() + n * kBand, band_.begin());
  std::copy(rhs_.begin(), rhs_.begin() + n, x_.begin());
  bool ok = cholesky(band_.data(), n);
  if (ok) cholesky_solve(band_.data(), n, x_.data());

  // The limits, as rows C x <= d: on the knots and limit_samples points of each segment, and then
  // wherever a dense check of the solution finds it outside (long segments can bulge out between
  // the fixed points), re-solving until it is inside everywhere. They include the start, which may be
  // outside (e.g. after the limits changed).
  bool within = true;
  if (ok && (std::isfinite(lower) || std::isfinite(upper)))
  {
    const double lo = std::min(lower, joint.position[0]), hi = std::max(upper, joint.position[0]);
    // p(t_k + s h) = fixed + coef . x over the unknowns; false if it has none (fixed values only).
    const auto basis = [&](std::size_t k, double s, double coef[6], long index[6], double & fixed)
    {
      const double h = joint.time[k + 1] - joint.time[k];
      const double scale[6] = {1, h, h * h, 1, h, h * h};
      double power = 1;
      bool any = false;
      fixed = 0;
      for (int l = 0; l < 6; ++l) coef[l] = 0;
      for (int i = 0; i < 6; ++i, power *= s)
        for (int l = 0; l < 6; ++l) coef[l] += kHermite[i][l] * power;
      for (int l = 0; l < 6; ++l)
      {
        const std::size_t knot = k + static_cast<std::size_t>(l / 3);
        coef[l] *= scale[l];
        index[l] = unknown(knot, l % 3);
        if (index[l] < 0)
          fixed += coef[l] * value(knot, l % 3);
        else
          any = any || coef[l] != 0.0;
      }
      return any;
    };
    std::size_t m = 0;
    const auto add_row = [&](std::size_t k, double s, double sign)
    {
      double coef[6], fixed;
      long index[6];
      const double bound = sign > 0 ? hi : -lo;
      if (!std::isfinite(bound) || !basis(k, s, coef, index, fixed) || m == row_bound_.size()) return;
      for (int l = 0; l < 6; ++l)
      {
        row_coef_[6 * m + l] = index[l] < 0 ? 0.0 : sign * coef[l];
        row_index_[6 * m + l] = index[l];
      }
      dual_[m] = std::numeric_limits<double>::quiet_NaN();
      row_bound_[m++] = bound - sign * fixed;
    };
    const std::size_t samples = options_.limit_samples;
    for (std::size_t k = 0; k < last; ++k)
    {
      for (std::size_t q = 1; q <= samples + 1; ++q)
      {
        const double s = static_cast<double>(q) / static_cast<double>(samples + 1);
        add_row(k, s, 1.0);
        add_row(k, s, -1.0);
      }
    }
    for (std::size_t round = 0;; ++round)
    {
      double worst = 0;
      for (std::size_t r = 0; r < m; ++r) worst = std::max(worst, row_value(r));
      // Each segment's extrema (where its velocity changes sign: on a grid, then bisected), as rows
      // where they are outside.
      bool added = false;
      for (std::size_t k = 0; k < last; ++k)
      {
        const double h = joint.time[k + 1] - joint.time[k];
        const double scale[6] = {1, h, h * h, 1, h, h * h};
        double c[6] = {};
        for (int l = 0; l < 6; ++l)
        {
          const std::size_t knot = k + static_cast<std::size_t>(l / 3);
          const long i = unknown(knot, l % 3);
          const double x = scale[l] * (i < 0 ? value(knot, l % 3) : x_[static_cast<std::size_t>(i)]);
          for (int q = 0; q < 6; ++q) c[q] += kHermite[q][l] * x;
        }
        const auto position = [&](double u) { return ((((c[5] * u + c[4]) * u + c[3]) * u + c[2]) * u + c[1]) * u + c[0]; };
        const auto velocity = [&](double u) { return (((5 * c[5] * u + 4 * c[4]) * u + 3 * c[3]) * u + 2 * c[2]) * u + c[1]; };
        double u0 = 0, v0 = velocity(0);
        for (std::size_t q = 1; q <= kDenseChecks; ++q)
        {
          const double u1 = static_cast<double>(q) / static_cast<double>(kDenseChecks), v1 = velocity(u1);
          if ((v0 > 0) != (v1 > 0) && q < kDenseChecks + 1)
          {
            double a = u0, b = u1, va = v0;
            for (int it = 0; it < 40; ++it)
            {
              const double mid = 0.5 * (a + b), vm = velocity(mid);
              if ((vm > 0) == (va > 0)) a = mid, va = vm;
              else b = mid;
            }
            const double u = 0.5 * (a + b), p = position(u);
            if (p - hi > kLimitTolerance && m < row_bound_.size()) add_row(k, u, 1.0), added = true;
            if (lo - p > kLimitTolerance && m < row_bound_.size()) add_row(k, u, -1.0), added = true;
          }
          u0 = u1, v0 = v1;
        }
      }
      if (worst <= kLimitTolerance && !added) break;
      if (round == kLimitRounds || !solve_limited(n, m))
      {
        within = false;
        break;
      }
    }
  }

  if (!ok)
  {
    // Not expected (the weights make H positive definite); fall back to stopping at each target.
    for (std::size_t k = 1; k <= last; ++k)
    {
      joint.position[k] = std::isnan(joint.target[k]) ? joint.target[k + 1] : joint.target[k];
      joint.velocity[k] = joint.acceleration[k] = 0.0;
    }
    return true;
  }
  for (std::size_t k = 1; k < last; ++k)
  {
    joint.position[k] = x_[3 * (k - 1)];
    joint.velocity[k] = x_[3 * (k - 1) + 1];
    joint.acceleration[k] = x_[3 * (k - 1) + 2];
  }
  if (free_end) joint.velocity[last] = x_[n - 1];
  return within;
}

double TrajectoryBuffer::row_value(std::size_t r) const
{
  double sum = -row_bound_[r];
  for (int l = 0; l < 6; ++l)
  {
    const auto i = row_index_[6 * r + l];
    if (i >= 0) sum += row_coef_[6 * r + l] * x_[static_cast<std::size_t>(i)];
  }
  return sum;
}

void TrajectoryBuffer::swap_rows(std::size_t a, std::size_t b)
{
  for (int l = 0; l < 6; ++l)
  {
    std::swap(row_coef_[6 * a + l], row_coef_[6 * b + l]);
    std::swap(row_index_[6 * a + l], row_index_[6 * b + l]);
  }
  std::swap(row_bound_[a], row_bound_[b]);
  std::swap(dual_[a], dual_[b]);
}

// The QP with only the rows that matter: those violated or near their bound at the unconstrained
// minimum, plus any the solution then violates (re-solving from it), until none is.
bool TrajectoryBuffer::solve_limited(std::size_t n, std::size_t m)
{
  constexpr double kNear = 1e-2;  // rad
  constexpr std::size_t kRounds = 8;
  std::size_t working = 0;
  for (std::size_t r = 0; r < m; ++r)
  {
    if (row_value(r) > -kNear) swap_rows(r, working++);
  }
  for (std::size_t round = 0; round < kRounds; ++round)
  {
    if (!solve_qp(n, working)) return false;
    const std::size_t before = working;
    for (std::size_t r = working; r < m; ++r)
    {
      if (row_value(r) > kLimitTolerance) swap_rows(r, working++);
    }
    if (working == before) return true;
  }
  double worst = 0;
  for (std::size_t r = 0; r < m; ++r) worst = std::max(worst, row_value(r));
  return worst <= kLimitTolerance;
}

// min 1/2 x^T H x - b^T x  s.t.  C x <= d, by a primal-dual interior-point method with Mehrotra's
// predictor-corrector: slacks s = d - C x >= 0 and multipliers z >= 0 with s_r z_r -> 0. Each Newton
// step solves (H + C^T diag(z / s) C) dx = r, which has H's band since a row only spans one segment.
// On failure x_ is left at the unconstrained minimum.
bool TrajectoryBuffer::solve_qp(std::size_t n, std::size_t m)
{
  double * x = x_.data();
  double * s = slack_.data();
  double * z = dual_.data();
  double * ds = dslack_.data();
  double * dz = ddual_.data();
  double * rp = rp_.data();
  double * rc = rc_.data();
  double * dx = dx_.data();
  double * rd = rd_.data();
  const double * coef = row_coef_.data();
  const std::int64_t * index = row_index_.data();
  const double * d = row_bound_.data();
  const double * b = rhs_.data();

  const auto row_dot = [&](std::size_t r, const double * v)
  {
    double sum = 0;
    for (int l = 0; l < 6; ++l)
      if (index[6 * r + l] >= 0) sum += coef[6 * r + l] * v[index[6 * r + l]];
    return sum;
  };
  const auto add_transpose = [&](std::size_t r, double v, double * out)
  {
    for (int l = 0; l < 6; ++l)
      if (index[6 * r + l] >= 0) out[index[6 * r + l]] += coef[6 * r + l] * v;
  };

  // Start from the unconstrained minimum with positive slacks.
  double b_norm = 0;
  for (std::size_t i = 0; i < n; ++i) b_norm = std::max(b_norm, std::abs(b[i]));
  // Multipliers from the last solve of this fit (warm start), 1e-2 for new rows.
  for (std::size_t r = 0; r < m; ++r)
  {
    s[r] = std::max(d[r] - row_dot(r, x), 1e-2);
    z[r] = std::isnan(z[r]) ? 1e-2 : std::max(z[r], 1e-4);
  }

  bool converged = false;
  constexpr std::size_t kMaxIterations = 60;
  std::size_t iteration = 0;
  for (; iteration < kMaxIterations; ++iteration)
  {
    // Residuals: rd = H x - b + C^T z, rp = C x + s - d, and the duality measure.
    band_multiply(hessian_.data(), n, x, rd);
    for (std::size_t i = 0; i < n; ++i) rd[i] -= b[i];
    double mu = 0, rp_norm = 0, rd_norm = 0;
    for (std::size_t r = 0; r < m; ++r)
    {
      add_transpose(r, z[r], rd);
      rp[r] = row_dot(r, x) + s[r] - d[r];
      rp_norm = std::max(rp_norm, std::abs(rp[r]));
      mu += s[r] * z[r];
    }
    mu /= static_cast<double>(m);
    for (std::size_t i = 0; i < n; ++i) rd_norm = std::max(rd_norm, std::abs(rd[i]));
    if (rp_norm < 1e-10 && rd_norm < 1e-9 * (1 + b_norm) && mu < 1e-12)
    {
      converged = true;
      break;
    }

    // Newton matrix.
    double * band = band_.data();
    std::copy(hessian_.begin(), hessian_.begin() + n * kBand, band);
    for (std::size_t r = 0; r < m; ++r)
    {
      const double weight = z[r] / s[r];
      for (int a = 0; a < 6; ++a)
      {
        const long i = index[6 * r + a];
        if (i < 0) continue;
        for (int c = 0; c < 6; ++c)
        {
          const long j = index[6 * r + c];
          if (j < 0 || j > i) continue;
          band[i * kBand + (i - j)] += weight * coef[6 * r + a] * coef[6 * r + c];
        }
      }
    }
    if (!cholesky(band, n)) break;

    // Step for the complementarity target rc (s z -> rc = s z - target): dx, then dz and ds.
    const auto step = [&]()
    {
      for (std::size_t i = 0; i < n; ++i) dx[i] = -rd[i];
      for (std::size_t r = 0; r < m; ++r) add_transpose(r, -(z[r] / s[r] * rp[r] - rc[r] / s[r]), dx);
      cholesky_solve(band, n, dx);
      for (std::size_t r = 0; r < m; ++r)
      {
        dz[r] = z[r] / s[r] * (row_dot(r, dx) + rp[r]) - rc[r] / s[r];
        ds[r] = -(rc[r] + s[r] * dz[r]) / z[r];
      }
    };
    const auto max_step = [&]()
    {
      double alpha = 1.0;
      for (std::size_t r = 0; r < m; ++r)
      {
        if (ds[r] < 0) alpha = std::min(alpha, -s[r] / ds[r]);
        if (dz[r] < 0) alpha = std::min(alpha, -z[r] / dz[r]);
      }
      return alpha;
    };

    // Predictor (affine scaling), then the centering-corrector step.
    for (std::size_t r = 0; r < m; ++r) rc[r] = s[r] * z[r];
    step();
    const double alpha_affine = max_step();
    double mu_affine = 0;
    for (std::size_t r = 0; r < m; ++r) mu_affine += (s[r] + alpha_affine * ds[r]) * (z[r] + alpha_affine * dz[r]);
    mu_affine /= static_cast<double>(m);
    const double sigma = std::pow(mu_affine / mu, 3);
    for (std::size_t r = 0; r < m; ++r) rc[r] = s[r] * z[r] + ds[r] * dz[r] - sigma * mu;
    step();
    const double alpha = std::min(1.0, 0.99 * max_step());
    for (std::size_t i = 0; i < n; ++i) x[i] += alpha * dx[i];
    for (std::size_t r = 0; r < m; ++r)
    {
      s[r] += alpha * ds[r];
      z[r] += alpha * dz[r];
    }
  }
  qp_iterations_ += iteration;

  // Accept the result if it meets the limits (to 1e-7) even if not fully converged.
  double worst = 0;
  for (std::size_t r = 0; r < m; ++r) worst = std::max(worst, row_dot(r, x) - d[r]);
  if (converged || worst <= kLimitTolerance) return true;
  // Failed (e.g. infeasible: the start moving out too fast to stop): the unconstrained minimum.
  std::copy(hessian_.begin(), hessian_.begin() + n * kBand, band_.begin());
  std::copy(rhs_.begin(), rhs_.begin() + n, x_.begin());
  if (cholesky(band_.data(), n)) cholesky_solve(band_.data(), n, x_.data());
  return false;
}

}  // namespace trajectory_buffer
