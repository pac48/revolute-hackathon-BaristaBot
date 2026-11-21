"""Position-servo version of a torque-actuated model, for planning controls as position targets.

The reBot's actuators are torque motors that a controller (a joint impedance controller:
tau = kp (q* - q) + kd (v* - v)) closes the loop around. The MPC plans the position targets q* the
controller is sent, so its model needs the actuators to be those servos.
"""

from __future__ import annotations

from typing import Mapping

import mujoco
import numpy as np


def position_servos(mjm: mujoco.MjModel, gains: Mapping[str, tuple[float, float]], disable_contact: bool = True,
                    gravity_compensated: bool = True) -> None:
    """Turns the actuators of the joints in `gains` (joint -> (kp, kd)) into position servos, in place.

    The new control is the joint's target position (its ctrlrange is the joint's range, +-pi if
    unlimited); the servo's force is kp (ctrl - q) - kd qdot, limited to the motor's old control range.
    Actuators of other joints are left alone. `disable_contact` turns off contacts in the model (much
    faster rollouts; no collision avoidance). `gravity_compensated` turns gravity off in the model: the
    reBot's joint impedance controller adds gravity feedforward (its `feedforward.gravity`), so a servo that
    sags under gravity is a worse model of it, and the plan would compensate a sag the real arm does not have
    (the tool then stays short of the goal by that much).
    """
    for a in range(mjm.nu):
        joint = int(mjm.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_JOINT, joint)
        if name not in gains:
            continue
        kp, kd = gains[name]
        old_range = mjm.actuator_ctrlrange[a].copy()
        old_limited = bool(mjm.actuator_ctrllimited[a])
        mjm.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
        mjm.actuator_gainprm[a, :] = 0.0
        mjm.actuator_gainprm[a, 0] = kp
        mjm.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_AFFINE
        mjm.actuator_biasprm[a, :] = 0.0
        mjm.actuator_biasprm[a, 1] = -kp
        mjm.actuator_biasprm[a, 2] = -kd
        limited = bool(mjm.jnt_limited[joint])
        mjm.actuator_ctrllimited[a] = 1
        mjm.actuator_ctrlrange[a] = mjm.jnt_range[joint] if limited else (-np.pi, np.pi)
        if old_limited:
            mjm.actuator_forcelimited[a] = 1
            mjm.actuator_forcerange[a] = old_range
    if gravity_compensated:
        mjm.opt.gravity[:] = 0.0
    if disable_contact:
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        # MuJoCo Warp still checks the collision options of a model with contacts off.
        mjm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD)
