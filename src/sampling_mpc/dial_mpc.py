"""DIAL-MPC (https://github.com/LeCAR-Lab/dial-mpc) on MuJoCo Warp.

Diffusion-inspired annealing for sampling-based MPC: each replan runs a few *diffusion* iterations,
each of which samples control-node perturbations around the current plan (with noise annealed over
both the iterations and the horizon: small near the present, large far ahead), rolls them all out
in parallel on the GPU, and moves the plan to the cost-weighted average (softmax of the standardized
reward). The plan is a few control nodes over the horizon, spline-interpolated to one control per
step, in [-1, 1] mapped onto the actuators' control ranges.

    cost = SiteCost(mjm, "tip", target=[0.4, 0, 0.3])
    mpc = DialMpc(mjm, cost, DialConfig(Nsample=1024, Hsample=16, Hnode=4))
    plan = mpc.plan(qpos, qvel)           # first call: Ndiffuse_init iterations
    apply(plan.controls[0]); mpc.shift()  # warm start the next plan
    plan = mpc.plan(qpos, qvel)

The differences from the reference: the simulator is MuJoCo Warp rather than Brax/MJX, the cost is a
`Cost` (costs.py, any number of terms) rather than an environment's reward, and the controls the plan
returns are in actuator units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from src.sampling_mpc.costs import Cost
from src.sampling_mpc.rollout import BatchedRollout
from src.sampling_mpc.splines import NodeSpline


@dataclass
class DialConfig:
    seed: int = 0
    ctrl_dt: float = 0.02  # time per control step, s: a multiple of the model's timestep
    Nsample: int = 2048  # number of samples
    Hsample: int = 16  # control steps in the horizon
    Hnode: int = 4  # intervals between control nodes (Hnode + 1 nodes)
    Ndiffuse: int = 2  # diffusion iterations per replan
    Ndiffuse_init: int = 10  # iterations of the first replan
    temp_sample: float = 0.06  # softmax temperature on the standardized reward
    horizon_diffuse_factor: float = 0.9  # noise falls by this per node toward the present
    traj_diffuse_factor: float = 0.5  # noise falls by this per diffusion iteration
    sigma_scale: float = 1.0  # noise scale of the last node, in normalized control units
    # The plan moves update_gain times the step to the weighted average of the samples (1: to it, as in the
    # reference). With small noise the average is close to the plan however good the samples are, so the
    # step is small and the plan slow to move; a gain above 1 takes a longer step along the same direction
    # (like a learning rate on the estimated gradient). The plan stays within [-1, 1].
    update_gain: float = 1.0
    # What `shift` appends at the end of the horizon: the last control ("hold"), or the middle of
    # each actuator's range ("zero", as in the reference, whose actions are centered on a default pose).
    shift_fill: str = "hold"
    # Normalized control 1 is this many actuator units for actuators without a control range.
    unbounded_ctrl_scale: float = 1.0


@dataclass
class Plan:
    controls: np.ndarray  # (Hsample, nu), actuator units; controls[0] is for now
    nodes: np.ndarray  # (Hnode + 1, nu), normalized to [-1, 1]
    ctrl_dt: float
    cost: float  # mean cost per step of the plan's controls (the last sample of the last iteration)
    iterations: int  # diffusion iterations this replan ran
    sample_costs: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))  # of the last iteration


def noise_schedule(config: DialConfig, iterations: int) -> np.ndarray:
    """Noise scale per iteration and node, (iterations, Hnode + 1): large far into the horizon and early on."""
    horizon = config.horizon_diffuse_factor ** np.arange(config.Hnode + 1)[::-1] * config.sigma_scale
    return horizon[None, :] * config.traj_diffuse_factor ** np.arange(iterations)[:, None]


def softmax_weights(costs: np.ndarray, reference_cost: float, temperature: float) -> np.ndarray:
    """MPPI weights: softmax of (reward - reference reward) / std(reward) / temperature, reward = -cost."""
    rewards = -np.asarray(costs, dtype=np.float64)
    std = rewards.std()
    logits = (rewards - (-reference_cost)) / (std if std > 1e-12 else 1.0) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


class DialMpc:
    """Sampling MPC for `mjm` (its actuators are the controls) with the given cost."""

    def __init__(self, mjm: mujoco.MjModel, cost: Cost, config: DialConfig | None = None, device=None):
        self.mjm = mjm
        self.cost = cost
        self.config = config or DialConfig()
        c = self.config
        if mjm.nu == 0:
            raise ValueError("the model has no actuators to control")
        substeps = round(c.ctrl_dt / mjm.opt.timestep)
        if substeps < 1 or abs(substeps * mjm.opt.timestep - c.ctrl_dt) > 1e-9:
            raise ValueError(f"ctrl_dt {c.ctrl_dt} is not a multiple of the model's timestep {mjm.opt.timestep}")
        if c.shift_fill not in ("hold", "zero"):
            raise ValueError("shift_fill is 'hold' or 'zero'")
        self.nu = mjm.nu
        self.substeps = substeps
        self.spline = NodeSpline(c.Hsample, c.Hnode)
        # Normalized control -> actuator units: center + scale * y.
        limited = mjm.actuator_ctrllimited.astype(bool)
        low, high = mjm.actuator_ctrlrange[:, 0], mjm.actuator_ctrlrange[:, 1]
        self.ctrl_center = np.where(limited, (low + high) / 2, 0.0)
        self.ctrl_scale = np.where(limited, (high - low) / 2, c.unbounded_ctrl_scale)
        self._device = device
        self._predictor = None  # for predict(), made when first needed
        self.rollout = BatchedRollout(mjm, cost, c.Nsample + 1, c.Hsample, substeps, device=device)
        self.rng = np.random.default_rng(c.seed)
        self.reset()

    def reset(self) -> None:
        """Forgets the plan (all controls at the middle of the ranges) and the warm start."""
        self.nodes = np.zeros((self.config.Hnode + 1, self.nu))
        self._first = True

    def hold(self, controls) -> None:
        """Sets the plan to hold `controls` (nu, actuator units) over the horizon: for a model whose controls
        are position targets, the current positions, so that the first plan does not start from mid-range."""
        self.nodes = np.tile(self.normalize(controls), (self.config.Hnode + 1, 1))

    def _to_actuator(self, normalized: np.ndarray) -> np.ndarray:
        return self.ctrl_center + self.ctrl_scale * normalized

    def normalize(self, controls) -> np.ndarray:
        """Controls in actuator units (..., nu) as plan nodes: normalized to [-1, 1] over the control ranges."""
        return np.clip((np.asarray(controls, dtype=float) - self.ctrl_center) / self.ctrl_scale, -1.0, 1.0)

    def _iteration(self, qpos, qvel, act, sigma: np.ndarray):
        """One reverse (denoising) step from the plan `self.nodes` at noise scales `sigma` (Hnode + 1)."""
        c = self.config
        nodes = self.nodes
        noise = self.rng.standard_normal((c.Nsample, c.Hnode + 1, self.nu))
        samples = nodes[None] + noise * sigma[None, :, None]
        samples[:, 0] = nodes[0]  # the control about to be applied is not perturbed
        samples = np.concatenate([samples, nodes[None]], axis=0)  # the current plan is a sample too
        samples = np.clip(samples, -1.0, 1.0)
        controls = self._to_actuator(self.spline.controls(samples))
        costs = self.rollout.run(qpos, qvel, controls, act) / c.Hsample  # mean cost per step
        weights = softmax_weights(costs, costs[-1], c.temp_sample)
        mean = np.einsum("n,nku->ku", weights, samples)
        self.nodes = np.clip(nodes + c.update_gain * (mean - nodes), -1.0, 1.0)
        return costs

    def plan(self, qpos, qvel, act=None, iterations: int | None = None) -> Plan:
        """Replans from the state (qpos, qvel[, act]) of the model, warm-started by the last plan.

        Runs `Ndiffuse` iterations (`Ndiffuse_init` the first time after construction or `reset`), or
        `iterations` if given."""
        c = self.config
        n = iterations if iterations is not None else (c.Ndiffuse_init if self._first else c.Ndiffuse)
        self._first = False
        costs = np.zeros(0)
        for sigma in noise_schedule(c, n):
            costs = self._iteration(qpos, qvel, act, sigma)
        controls = self._to_actuator(self.spline.controls(self.nodes))
        return Plan(controls, self.nodes.copy(), c.ctrl_dt, float(costs[-1]) if n else float("nan"), n, costs)

    def predict(self, qpos, qvel, controls=None, act=None) -> np.ndarray:
        """Joint positions (Hsample + 1, nq) of the model from the state under `controls` (Hsample, nu,
        actuator units; default: the current plan): at the start and after each control step.

        The plan's controls can be jumpy where the cost is indifferent; this is the smooth motion they
        produce, e.g. for a reference to a controller."""
        if self._predictor is None:
            c = self.config
            self._predictor = BatchedRollout(self.mjm, None, 1, c.Hsample, self.substeps, record_qpos=True,
                                             device=self._device)
        if controls is None:
            controls = self._to_actuator(self.spline.controls(self.nodes))
        self._predictor.run(qpos, qvel, np.asarray(controls)[None], act)
        return self._predictor.qpos_trace_numpy()[0]

    def shift(self, n_steps: int = 1) -> None:
        """Advances the plan by `n_steps` control steps (call after applying them), to warm start the next plan."""
        u = self.spline.controls(self.nodes)
        last = u[-1].copy()
        u = np.roll(u, -n_steps, axis=0)
        u[-n_steps:] = last if self.config.shift_fill == "hold" else 0.0
        self.nodes = self.spline.fit(u)
