"""Batched rollouts of a MuJoCo model on the GPU (MuJoCo Warp): many control sequences from one state."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

from src.sampling_mpc.costs import Cost


@wp.kernel
def _reset_kernel(
    qpos0: wp.array(dtype=wp.float32),
    qvel0: wp.array(dtype=wp.float32),
    act0: wp.array(dtype=wp.float32),
    qpos: wp.array2d(dtype=wp.float32),
    qvel: wp.array2d(dtype=wp.float32),
    act: wp.array2d(dtype=wp.float32),
    time: wp.array(dtype=wp.float32),
    cost: wp.array(dtype=wp.float32),
):
    w = wp.tid()
    for i in range(qpos0.shape[0]):
        qpos[w, i] = qpos0[i]
    for i in range(qvel0.shape[0]):
        qvel[w, i] = qvel0[i]
    for i in range(act0.shape[0]):
        act[w, i] = act0[i]
    time[w] = 0.0
    cost[w] = 0.0


@wp.kernel
def _apply_ctrl_kernel(
    controls: wp.array3d(dtype=wp.float32),
    step: int,
    ctrl: wp.array2d(dtype=wp.float32),
):
    w = wp.tid()
    for i in range(ctrl.shape[1]):
        ctrl[w, i] = controls[w, step, i]


@wp.kernel
def _record_kernel(
    qpos: wp.array2d(dtype=wp.float32),
    step: int,
    out: wp.array3d(dtype=wp.float32),
):
    w = wp.tid()
    for i in range(qpos.shape[1]):
        out[w, step, i] = qpos[w, i]


class BatchedRollout:
    """`nworld` simulations of `mjm` from one state, each with its own control sequence.

    A rollout resets every world to (qpos, qvel, act), then for each of `steps` control steps applies
    that step's control, advances the physics `substeps` times (the model's timestep each), and adds
    the cost's stage cost; the cost's terminal cost is added at the end. The whole rollout is
    captured as one CUDA graph on first use, as a launch per physics step would dominate otherwise.

    Contacts and the constraint solver are as in MuJoCo (`mjm`'s options). Not thread-safe.
    """

    def __init__(
        self,
        mjm: mujoco.MjModel,
        cost: Cost | None,
        nworld: int,
        steps: int,
        substeps: int,
        record_qpos: bool = False,
        device=None,
        **make_data_kwargs,
    ):
        self.nworld, self.steps, self.substeps = nworld, steps, substeps
        self.nq, self.nv, self.nu, self.na = mjm.nq, mjm.nv, mjm.nu, mjm.na
        self.cost = cost
        self.device = wp.get_device(device)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(mjm)
            self.d = mjw.make_data(mjm, nworld=nworld, **make_data_kwargs)
            self._qpos0 = wp.zeros(max(self.nq, 1), dtype=wp.float32)
            self._qvel0 = wp.zeros(max(self.nv, 1), dtype=wp.float32)
            self._act0 = wp.zeros(max(self.na, 0), dtype=wp.float32)
            self.controls = wp.zeros((nworld, steps, max(self.nu, 1)), dtype=wp.float32)
            self.costs = wp.zeros(nworld, dtype=wp.float32)
            self.qpos_trace = wp.zeros((nworld, steps + 1, self.nq), dtype=wp.float32) if record_qpos else None
        self._graph = None

    def _launch(self) -> None:
        d, m = self.d, self.m
        wp.launch(_reset_kernel, dim=self.nworld,
                  inputs=[self._qpos0, self._qvel0, self._act0, d.qpos, d.qvel, d.act, d.time, self.costs])
        if self.cost is not None:
            self.cost.begin(m, d)
        if self.qpos_trace is not None:
            wp.launch(_record_kernel, dim=self.nworld, inputs=[d.qpos, 0, self.qpos_trace])
        for t in range(self.steps):
            wp.launch(_apply_ctrl_kernel, dim=self.nworld, inputs=[self.controls, t, d.ctrl])
            for _ in range(self.substeps):
                mjw.step(m, d)
            if self.cost is not None:
                self.cost.running(m, d, self.costs)
            if self.qpos_trace is not None:
                wp.launch(_record_kernel, dim=self.nworld, inputs=[d.qpos, t + 1, self.qpos_trace])
        if self.cost is not None:
            self.cost.terminal(m, d, self.costs)

    def run(self, qpos: np.ndarray, qvel: np.ndarray, controls: np.ndarray, act: np.ndarray | None = None) -> np.ndarray:
        """Rolls out `controls` (nworld, steps, nu; actuator units) from the state; returns costs (nworld,)."""
        with wp.ScopedDevice(self.device):
            self._qpos0.assign(np.asarray(qpos, dtype=np.float32).reshape(-1))
            self._qvel0.assign(np.asarray(qvel, dtype=np.float32).reshape(-1))
            if self.na:
                self._act0.assign(np.zeros(self.na, np.float32) if act is None else np.asarray(act, np.float32))
            ctrl = np.zeros((self.nworld, self.steps, max(self.nu, 1)), dtype=np.float32)
            ctrl[:, :, : self.nu] = controls
            self.controls.assign(ctrl)
            if self._graph is None:
                self._launch()  # first run compiles the kernels (and warms up), outside the capture
                with wp.ScopedCapture() as capture:
                    self._launch()
                self._graph = capture.graph
            wp.capture_launch(self._graph)
            return self.costs.numpy().copy()

    def qpos_trace_numpy(self) -> np.ndarray:
        """(nworld, steps + 1, nq): positions at the start and after each control step of the last run."""
        if self.qpos_trace is None:
            raise RuntimeError("created without record_qpos")
        return self.qpos_trace.numpy().copy()
