# sampling_mpc

Sampling-based MPC for trajectory generation on the GPU: [DIAL-MPC](https://github.com/LeCAR-Lab/dial-mpc)
(diffusion-inspired annealing for MPC) implemented on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp),
with pluggable cost functions. Python; deps are the `@pip_mpc` hub (`requirements.in`).

```python
from src.sampling_mpc.costs import SiteCost, StateCost, SumCost
from src.sampling_mpc.dial_mpc import DialConfig, DialMpc

cost = SumCost(
    SiteCost(mjm, "tip", target=[0.45, 0.46, 0.17], weight=100.0, terminal_scale=5.0),
    StateCost(mjm, qvel_weight=0.01, ctrl_weight=0.001),
)
mpc = DialMpc(mjm, cost, DialConfig(Nsample=1024, Hsample=16, Hnode=4))

while True:
    plan = mpc.plan(data.qpos, data.qvel)   # first call: Ndiffuse_init iterations, then Ndiffuse
    data.ctrl[:] = plan.controls[0]          # actuator units; plan.controls is (Hsample, nu)
    ...step the real system / simulation for plan.ctrl_dt...
    mpc.shift()                              # warm start the next plan
```

## How it works

Each `plan()` runs a few diffusion iterations. One iteration samples `Nsample` perturbations of the
plan's control nodes (`Hnode + 1` nodes over the horizon, interpolated to `Hsample` controls by a
quadratic spline, normalized to [-1, 1] over the actuators' control ranges), rolls all of them (and
the current plan) out in parallel with MuJoCo Warp, and moves the plan to the softmax-weighted
average of the samples (weights from the standardized reward, `temp_sample`). The noise is annealed
over the horizon (`horizon_diffuse_factor`: small near the present) and over the iterations
(`traj_diffuse_factor`). The control about to be applied is never perturbed. It follows the reference
(`dial_core.py`: `reverse_once`, the sigma schedule, `shift`) with these differences: MuJoCo Warp instead of
Brax/MJX; a `Cost` instead of an environment's reward; controls returned in actuator units;
`DialConfig.shift_fill` (`"hold"` by default; `"zero"` is the reference's).

A rollout is one CUDA graph (reset, every control step, the physics substeps, the cost), captured on
the first `plan()`; that first call also compiles the Warp kernels (tens of seconds, cached after).
Roughly 85 ms per replan (2 iterations, 1025 worlds) for a two-link arm on an RTX 4090.

## Cost functions

A `Cost` adds to a per-world cost array on the GPU after every control step (`running`) and after the
last one (`terminal`). Built in (`costs.py`):

- `StateCost`: quadratic cost on joint positions (to targets), velocities, and controls.
- `SiteCost`: squared distance of a site to a point; `set_target()` moves it between plans.
- `SitePoseCost`: a site's position and orientation (`|R - R*|_F^2`); `set_target()` likewise.
- `SumCost`: combines costs.

A new cost is a subclass launching a Warp kernel on MuJoCo Warp's batched `Data`
(`d.xpos[world, body]`, `d.site_xpos`, `d.qpos`, `d.sensordata`, ...):

```python
@wp.kernel
def _height_kernel(xpos: wp.array2d(dtype=wp.vec3f), body: int, target: float, cost: wp.array(dtype=wp.float32)):
    w = wp.tid()
    e = xpos[w, body][2] - target
    cost[w] += e * e

class HeightCost(Cost):
    def __init__(self, mjm, body, target):
        self.body = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, body)
        self.target = target
    def running(self, m, d, cost):
        wp.launch(_height_kernel, dim=d.nworld, inputs=[d.xpos, self.body, self.target, cost])
```

Values that change between plans must live in device arrays (like `SiteCost`'s target): the rollout
graph bakes in kernel arguments.

## Position-controlled arms

The controls are the model's actuators. For a torque-motor arm driven by a position controller,
`servo.position_servos(mjm, {joint: (kp, kd)})` turns its motors into position servos (the controller's
gains, force limited to the motors'), so the plan is joint position targets. Then:

- `mpc.hold(positions)` sets the plan to hold the current joint positions (call before the first plan; the
  default plan is the middle of the control ranges).
- Use a small `sigma_scale` (about 0.15) for position targets: the reference's 1.0 is for normalized
  actions around a default pose, and here means +-half the joint range.
- `mpc.predict(qpos, qvel, plan.controls)` is the motion the model makes under the plan. It is smoother than
  the plan's targets, but a controller tracking it sags below it (gravity) a second time, so send the targets.
- A position servo that sags under gravity is a bad model of a controller with gravity feedforward: the plan
  compensates a sag the real arm does not have, and the tool stays short. `position_servos` turns gravity off.
- Do not send the plan to a controller as it is: its targets are jagged at the scale of the control nodes and every
  replan differs from the last. `trajectory_generator.mpc_reaching.QuinticSetpoint` moves a setpoint smoothly to where
  the plan ends and restarts from its own position, velocity and acceleration at every replan, so it never jumps.
  (On the reBot this took the sent reference from 6-16 rad/s^2 RMS to about 1, with no content above 5 Hz; the
  controller's own noise floor is about 3-4, measured with `//src/trajectory_generator:shake_diagnostic`.)

`//src/trajectory_generator:mpc_reach` does this on the reBot, in real time against the running controller.

## Files

| | |
| --- | --- |
| `dial_mpc.py` | `DialMpc`, `DialConfig`, `Plan`, `noise_schedule`, `softmax_weights` |
| `costs.py` | `Cost`, `StateCost`, `SiteCost`, `SumCost` |
| `rollout.py` | `BatchedRollout`: parallel rollouts of a model from one state (CUDA graph) |
| `splines.py` | `NodeSpline`: control nodes <-> per-step controls |
| `servo.py` | `position_servos`: torque motors -> position servos with the controller's gains |

## Tests

```
bazel test //src/sampling_mpc:splines_test //src/sampling_mpc:dial_core_test          # no GPU
bazel test //src/sampling_mpc:dial_mpc_gpu_test                                       # NVIDIA GPU, unsandboxed
```

`dial_mpc_gpu_test` closes the loop on a two-link arm with a site cost and a joint-target cost, and
is tagged `manual`/`local`/`requires-gpu` so `bazel test //...` skips it.

## Limits

- Controls are the model's actuators; `ctrl_dt` must be a multiple of the model's timestep.
- Costs and the plan only see joint space states (`qpos`, `qvel`, `act`); a cost that needs more
  reads it from `d` in its kernel.
- No hardware-in-the-loop interface yet: `plan()` takes `qpos`/`qvel` (e.g. from a `StateEstimator`).
