"""Cost functions of the sampling MPC, evaluated on the GPU alongside the simulation.

A rollout calls `begin(...)` after resetting every world to the start state, adds `running(...)` after
every control step and `terminal(...)` after the last one, to a per-world cost array (`cost[world]`, float32, lower is better). Costs are Warp kernels over
MuJoCo Warp's batched `Data` (`d.qpos[world, i]`, `d.site_xpos[world, i]`, ...), so a new cost is a
subclass that launches its own kernel(s):

    class UprightCost(Cost):
        def running(self, m, d, cost):
            wp.launch(_upright_kernel, dim=d.nworld, inputs=[d.xquat, self.body, cost])

Built-in: StateCost (joint positions, velocities and controls), SiteCost (a site's position),
SitePoseCost (its position and orientation), EffortCost (actuator forces and how fast the controls change: to
keep motion smooth), VelocityLimitCost (joint velocity limits), GroundCost (the robot above the floor), and SumCost to combine them.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import mujoco
import numpy as np
import warp as wp


class Cost:
    """Base of the cost functions. `running` and `terminal` add to `cost` (wp.array of float32, one per world)."""

    def begin(self, m, d) -> None:
        """Called at the start of every rollout, after the worlds are reset (for costs with state across steps)."""

    def running(self, m, d, cost) -> None:
        """Stage cost of the state in `d` (after one control step), added to `cost`."""

    def terminal(self, m, d, cost) -> None:
        """Cost of the final state of the rollout, added to `cost`. Defaults to nothing."""


@wp.kernel
def _state_cost_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    q_target: wp.array(dtype=wp.float32),
    q_weight: wp.array(dtype=wp.float32),
    v_weight: wp.array(dtype=wp.float32),
    u_target: wp.array(dtype=wp.float32),
    u_weight: wp.array(dtype=wp.float32),
    scale: float,
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    total = float(0.0)
    for i in range(q_target.shape[0]):
        e = qpos[w, i] - q_target[i]
        total += q_weight[i] * e * e
    for i in range(v_weight.shape[0]):
        total += v_weight[i] * qvel[w, i] * qvel[w, i]
    for i in range(u_weight.shape[0]):
        e = ctrl[w, i] - u_target[i]
        total += u_weight[i] * e * e
    cost[w] += scale * total


@wp.kernel
def _site_cost_kernel(
    site_xpos: wp.array2d(dtype=wp.vec3f),
    site: int,
    target: wp.array(dtype=wp.vec3f),
    weight: float,
    scale: float,
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    e = site_xpos[w, site] - target[0]
    cost[w] += scale * weight * wp.dot(e, e)


_SCALAR_JOINTS = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))


def _per_joint(mjm: mujoco.MjModel, values: Mapping[str, float] | float | None, size: int, address, what: str):
    """A length-`size` vector from a joint name -> value mapping (or one value for all joints)."""
    out = np.zeros(size, dtype=np.float32)
    if values is None:
        return out
    joints = range(mjm.njnt)
    if isinstance(values, Mapping):
        for name, value in values.items():
            jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or int(mjm.jnt_type[jid]) not in _SCALAR_JOINTS:
                raise ValueError(f"{what}: '{name}' is not a hinge or slide joint of the model")
            out[address(jid)] = value
    else:
        for jid in joints:
            if int(mjm.jnt_type[jid]) in _SCALAR_JOINTS:
                out[address(jid)] = values
    return out


class StateCost(Cost):
    """Quadratic cost on joint positions, joint velocities and controls.

    stage = sum_i qpos_weight[i] (qpos_i - qpos_target[i])^2 + qvel_weight[i] qvel_i^2
            + ctrl_weight[a] (ctrl_a - ctrl_target[a])^2
    The terminal cost is the same with `terminal_scale` times the weights (position and velocity terms
    only, there is no control at the end).

    Weights are joint name -> weight, or one number for every hinge/slide joint (qpos and qvel only);
    targets are joint name -> value (default 0, and for controls the actuator's own value: 0).
    `ctrl_weight` is one number for every actuator, or actuator name -> weight.
    """

    def __init__(
        self,
        mjm: mujoco.MjModel,
        qpos_target: Mapping[str, float] | None = None,
        qpos_weight: Mapping[str, float] | float | None = None,
        qvel_weight: Mapping[str, float] | float | None = None,
        ctrl_weight: Mapping[str, float] | float | None = None,
        ctrl_target: Mapping[str, float] | None = None,
        terminal_scale: float = 1.0,
        device=None,
    ):
        qadr = lambda j: mjm.jnt_qposadr[j]
        dadr = lambda j: mjm.jnt_dofadr[j]
        self._q_target = wp.array(_per_joint(mjm, qpos_target, mjm.nq, qadr, "qpos_target"), dtype=wp.float32, device=device)
        self._q_weight = wp.array(_per_joint(mjm, qpos_weight, mjm.nq, qadr, "qpos_weight"), dtype=wp.float32, device=device)
        self._v_weight = wp.array(_per_joint(mjm, qvel_weight, mjm.nv, dadr, "qvel_weight"), dtype=wp.float32, device=device)

        def per_actuator(values, what):
            out = np.zeros(mjm.nu, dtype=np.float32)
            if values is None:
                return out
            if isinstance(values, Mapping):
                for name, value in values.items():
                    aid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                    if aid < 0:
                        raise ValueError(f"{what}: '{name}' is not an actuator of the model")
                    out[aid] = value
            else:
                out[:] = values
            return out

        self._u_weight = wp.array(per_actuator(ctrl_weight, "ctrl_weight"), dtype=wp.float32, device=device)
        self._u_target = wp.array(per_actuator(ctrl_target, "ctrl_target"), dtype=wp.float32, device=device)
        self._none_u = wp.zeros(0, dtype=wp.float32, device=device)
        self._terminal_scale = float(terminal_scale)

    def running(self, m, d, cost) -> None:
        wp.launch(
            _state_cost_kernel,
            dim=cost.shape[0],
            inputs=[d.qpos, d.qvel, d.ctrl, self._q_target, self._q_weight, self._v_weight,
                    self._u_target, self._u_weight, 1.0, cost],
        )

    def terminal(self, m, d, cost) -> None:
        wp.launch(
            _state_cost_kernel,
            dim=cost.shape[0],
            inputs=[d.qpos, d.qvel, d.ctrl, self._q_target, self._q_weight, self._v_weight,
                    self._u_target, self._none_u, self._terminal_scale, cost],
        )


class SiteCost(Cost):
    """Squared distance of a site to a target point (world frame): weight |site - target|^2."""

    def __init__(self, mjm: mujoco.MjModel, site: str, target: Sequence[float], weight: float = 1.0,
                 terminal_scale: float = 1.0, device=None):
        self._site = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, site)
        if self._site < 0:
            raise ValueError(f"site '{site}' is not in the model")
        # On the device, so that moving it needs no new capture of the rollout.
        self._target = wp.array([wp.vec3f(*[float(x) for x in target])], dtype=wp.vec3f, device=device)
        self.weight = float(weight)
        self._terminal_scale = float(terminal_scale)

    def set_target(self, target: Sequence[float]) -> None:
        """Moves the target; takes effect at the next plan."""
        self._target.assign(np.asarray(target, dtype=np.float32).reshape(1, 3))

    def running(self, m, d, cost) -> None:
        wp.launch(_site_cost_kernel, dim=cost.shape[0],
                  inputs=[d.site_xpos, self._site, self._target, self.weight, 1.0, cost])

    def terminal(self, m, d, cost) -> None:
        wp.launch(_site_cost_kernel, dim=cost.shape[0],
                  inputs=[d.site_xpos, self._site, self._target, self.weight, self._terminal_scale, cost])


@wp.kernel
def _site_pose_cost_kernel(
    site_xpos: wp.array2d(dtype=wp.vec3f),
    site_xmat: wp.array2d(dtype=wp.mat33f),
    site: int,
    target: wp.array(dtype=wp.vec3f),
    target_rot: wp.array(dtype=wp.mat33f),
    pos_weight: float,
    rot_weight: float,
    scale: float,
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    e = site_xpos[w, site] - target[0]
    r = site_xmat[w, site] - target_rot[0]
    # |R - R*|_F^2 = 2 (3 - trace(R*^T R)): 0 aligned, 8 at most.
    rot = float(0.0)
    for i in range(3):
        for j in range(3):
            rot += r[i, j] * r[i, j]
    cost[w] += scale * (pos_weight * wp.dot(e, e) + rot_weight * rot)


class SitePoseCost(Cost):
    """Pose of a site (position and orientation) against a target: pos_weight |p - p*|^2 + rot_weight |R - R*|_F^2.

    `target_xmat` is the site's rotation matrix in the world frame, 9 numbers row-major (MuJoCo's
    `site_xmat`). |R - R*|_F^2 = 2 (3 - trace(R*^T R)) is 0 at the target orientation and, for a
    rotation by `angle` away, 8 sin^2(angle / 2): 8 at most.
    """

    def __init__(self, mjm: mujoco.MjModel, site: str, target_pos: Sequence[float], target_xmat: Sequence[float],
                 pos_weight: float = 1.0, rot_weight: float = 0.0, terminal_scale: float = 1.0, device=None):
        self._site = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_SITE, site)
        if self._site < 0:
            raise ValueError(f"site '{site}' is not in the model")
        self._target = wp.zeros(1, dtype=wp.vec3f, device=device)
        self._target_rot = wp.zeros(1, dtype=wp.mat33f, device=device)
        self.pos_weight = float(pos_weight)
        self.rot_weight = float(rot_weight)
        self._terminal_scale = float(terminal_scale)
        self.set_target(target_pos, target_xmat)

    def set_target(self, target_pos: Sequence[float], target_xmat: Sequence[float]) -> None:
        """Moves the target; takes effect at the next plan (no new capture of the rollout)."""
        self._target.assign(np.asarray(target_pos, dtype=np.float32).reshape(1, 3))
        self._target_rot.assign(np.asarray(target_xmat, dtype=np.float32).reshape(1, 3, 3))

    def _launch(self, d, cost, scale: float) -> None:
        wp.launch(_site_pose_cost_kernel, dim=cost.shape[0],
                  inputs=[d.site_xpos, d.site_xmat, self._site, self._target, self._target_rot,
                          self.pos_weight, self.rot_weight, scale, cost])

    def running(self, m, d, cost) -> None:
        self._launch(d, cost, 1.0)

    def terminal(self, m, d, cost) -> None:
        self._launch(d, cost, self._terminal_scale)


@wp.kernel
def _effort_cost_kernel(
    actuator_force: wp.array2d(dtype=wp.float32),
    ctrl: wp.array2d(dtype=wp.float32),
    force_weight: wp.array(dtype=wp.float32),
    rate_weight: wp.array(dtype=wp.float32),
    prev_ctrl: wp.array2d(dtype=wp.float32),
    prev_valid: wp.array(dtype=wp.int32),
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    total = float(0.0)
    for a in range(force_weight.shape[0]):
        total += force_weight[a] * actuator_force[w, a] * actuator_force[w, a]
        if prev_valid[w] == 1:
            e = ctrl[w, a] - prev_ctrl[w, a]
            total += rate_weight[a] * e * e
        prev_ctrl[w, a] = ctrl[w, a]
    prev_valid[w] = 1
    cost[w] += total


@wp.kernel
def _effort_begin_kernel(prev_valid: wp.array(dtype=wp.int32)):
    prev_valid[wp.tid()] = 0


class EffortCost(Cost):
    """Minimum effort: a penalty on the actuators' forces and on how fast the controls change.

        stage = sum_a (force_weight (f_a / f_scale_a)^2 + rate_weight (u_a,k - u_a,k-1)^2)

    `f_scale_a` is the actuator's force limit (1 if it has none), so `force_weight` is per (fraction of the
    limit)^2. The rate term (not on the first control step: nothing precedes it) is what keeps position
    targets from jittering from step to step, which a controller then turns into shaking. `rate_weight` is per
    (control units per step)^2: for position targets in rad, 100 makes a 0.05 rad step cost as much as a
    5 cm error at weight 100 on |p|^2 does.
    """

    def __init__(self, mjm: mujoco.MjModel, force_weight: float = 0.0, rate_weight: float = 0.0, device=None):
        scale = np.where(mjm.actuator_forcelimited.astype(bool), np.maximum(mjm.actuator_forcerange[:, 1], 1e-9), 1.0)
        self._force_weight = wp.array(force_weight / scale**2, dtype=wp.float32, device=device)
        self._rate_weight = wp.array(np.full(mjm.nu, rate_weight), dtype=wp.float32, device=device)
        self._device = device
        self._prev_ctrl = None
        self._prev_valid = None

    def _allocate(self, d) -> None:
        if self._prev_ctrl is None or self._prev_ctrl.shape[0] != d.nworld:
            with wp.ScopedDevice(self._device if self._device is not None else d.qpos.device):
                self._prev_ctrl = wp.zeros((d.nworld, self._force_weight.shape[0]), dtype=wp.float32)
                self._prev_valid = wp.zeros(d.nworld, dtype=wp.int32)

    def begin(self, m, d) -> None:
        self._allocate(d)
        wp.launch(_effort_begin_kernel, dim=d.nworld, inputs=[self._prev_valid])

    def running(self, m, d, cost) -> None:
        self._allocate(d)
        wp.launch(_effort_cost_kernel, dim=d.nworld,
                  inputs=[d.actuator_force, d.ctrl, self._force_weight, self._rate_weight, self._prev_ctrl,
                          self._prev_valid, cost])


@wp.kernel
def _velocity_limit_cost_kernel(
    qvel: wp.array2d(dtype=wp.float32),
    address: wp.array(dtype=wp.int32),
    limit: wp.array(dtype=wp.float32),
    weight: float,
    scale: float,
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    total = float(0.0)
    for i in range(address.shape[0]):
        e = wp.max(wp.abs(qvel[w, address[i]]) - limit[i], 0.0)
        total += e * e
    cost[w] += scale * weight * total


class VelocityLimitCost(Cost):
    """Joint velocity limits: nothing within them, steeply quadratic past them.

    stage = weight sum_j max(0, |qvel_j| - limit_j)^2
    over the hinge/slide joints in `joints` (default: all of them), in every running step and at the end
    (times `terminal_scale`). `limit` is rad/s (m/s for slides): one number for all, or joint name -> limit.
    """

    def __init__(self, mjm: mujoco.MjModel, joints: Sequence[str] | None = None,
                 limit: Mapping[str, float] | float = 1.0, weight: float = 1000.0, terminal_scale: float = 1.0,
                 device=None):
        ids = range(mjm.njnt) if joints is None else [mjm.joint(name).id for name in joints]
        ids = [j for j in ids if int(mjm.jnt_type[j]) in _SCALAR_JOINTS]
        name = lambda j: mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, j)
        limits = [limit[name(j)] if isinstance(limit, Mapping) else limit for j in ids]
        self._address = wp.array(np.array([mjm.jnt_dofadr[j] for j in ids], dtype=np.int32), dtype=wp.int32, device=device)
        self._limit = wp.array(np.array(limits, dtype=np.float32), dtype=wp.float32, device=device)
        self.weight = float(weight)
        self._terminal_scale = float(terminal_scale)

    def _launch(self, d, cost, scale: float) -> None:
        wp.launch(_velocity_limit_cost_kernel, dim=cost.shape[0],
                  inputs=[d.qvel, self._address, self._limit, self.weight, scale, cost], device=cost.device)

    def running(self, m, d, cost) -> None:
        self._launch(d, cost, 1.0)

    def terminal(self, m, d, cost) -> None:
        self._launch(d, cost, self._terminal_scale)


@wp.kernel
def _ground_cost_kernel(
    geom_xpos: wp.array2d(dtype=wp.vec3f),
    geom_xmat: wp.array2d(dtype=wp.mat33f),
    geoms: wp.array(dtype=wp.int32),
    center: wp.array(dtype=wp.vec3f),
    half: wp.array(dtype=wp.vec3f),
    height: float,
    weight: float,
    scale: float,
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    total = float(0.0)
    for i in range(geoms.shape[0]):
        g = geoms[i]
        R = geom_xmat[w, g]
        c = R * center[i]
        h = half[i]
        # The lowest corner of the geom's bounding box (in its own frame), in world z.
        low = geom_xpos[w, g][2] + c[2] - (wp.abs(R[2, 0]) * h[0] + wp.abs(R[2, 1]) * h[1] + wp.abs(R[2, 2]) * h[2])
        e = wp.max(height - low, 0.0)
        total += e * e
    cost[w] += scale * weight * total


class GroundCost(Cost):
    """The robot kept above the ground plane: nothing while it is, steeply quadratic below it.

    stage = weight sum_g max(0, height - lowest_g)^2
    over the collision geoms (contype or conaffinity set) of the bodies under `root_body` that move (not welded to
    the world: a base resting on the floor costs nothing), lowest_g the lowest corner of the geom's bounding box
    (MuJoCo's geom_aabb, in the geom's frame) in world z. For rollouts without contacts, where nothing else stops
    the arm going through the floor. Every running step and the end (times `terminal_scale`).
    """

    def __init__(self, mjm: mujoco.MjModel, root_body: str, height: float = 0.0, weight: float = 1e5,
                 terminal_scale: float = 1.0, device=None):
        root = mjm.body_rootid[mjm.body(root_body).id]
        geoms = [g for g in range(mjm.ngeom)
                 if (mjm.geom_contype[g] or mjm.geom_conaffinity[g])
                 and mjm.body_rootid[mjm.geom_bodyid[g]] == root and mjm.body_weldid[mjm.geom_bodyid[g]] != 0]
        self.geoms = geoms
        self._geoms = wp.array(np.array(geoms, dtype=np.int32), dtype=wp.int32, device=device)
        aabb = np.asarray(mjm.geom_aabb[geoms], dtype=np.float32).reshape(-1, 6)
        self._center = wp.array(aabb[:, :3], dtype=wp.vec3f, device=device)
        self._half = wp.array(aabb[:, 3:], dtype=wp.vec3f, device=device)
        self.height = float(height)
        self.weight = float(weight)
        self._terminal_scale = float(terminal_scale)

    def _launch(self, d, cost, scale: float) -> None:
        wp.launch(_ground_cost_kernel, dim=cost.shape[0],
                  inputs=[d.geom_xpos, d.geom_xmat, self._geoms, self._center, self._half, self.height, self.weight,
                          scale, cost], device=cost.device)

    def running(self, m, d, cost) -> None:
        self._launch(d, cost, 1.0)

    def terminal(self, m, d, cost) -> None:
        self._launch(d, cost, self._terminal_scale)


class SumCost(Cost):
    """The sum of costs."""

    def __init__(self, *costs: Cost):
        self.costs = costs

    def begin(self, m, d) -> None:
        for c in self.costs:
            c.begin(m, d)

    def running(self, m, d, cost) -> None:
        for c in self.costs:
            c.running(m, d, cost)

    def terminal(self, m, d, cost) -> None:
        for c in self.costs:
            c.terminal(m, d, cost)
