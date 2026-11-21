# Fixes to upstream to fast_robot_kinematics

This repo uses [fast_robot_kinematics](https://github.com/pac48/fast_robot_kinematics) at
`43395339e0cb02ffcc263a19288b1494f9662cdc` (`http_archive` in `MODULE.bazel`): its code template
(`code_generation/robot_config.cpp.template`), headers and the LBFGSpp IK. Upstream's generator,
`code_generation/robot_gen.py`, is replaced here by
`src/robot_model/robot_model_utils/fast_kinematics_gen.py`, because it gives wrong poses for the reBot
arm. The changes below should go upstream; once they have, the local generator can go
and `fast_kinematics.bzl` can call upstream's.

Each fix is covered by `//src/robot_model/robot_model_utils:fast_kinematics_gen_test`, which checks the
generated chain against direct URDF forward kinematics. `//src/robot_model/rebot_arm:rebot_arm_kinematics_test`
checks FK against MuJoCo and runs IK round trips.

## Generator (`code_generation/robot_gen.py`): wrong poses

1. **Joint axes are ignored.** The template always rotates about the joint frame's +z, but URDF joints
   can have any `<axis>`. The reBot's are `0 0 -1` and `0 0 1`; the test URDFs happen to use `0 0 1`
   everywhere.
   *Fix:* rotate each joint's frame by a constant A_i with A_i z = axis_i, since
   `origin_i * Rot(axis_i, q) = origin_i * A_i * Rot_z(q) * A_i^T`. Fold A_i into the joint's segment
   and A_i^T into the next one. The template stays the same. See `align_z` and `segments` in
   `fast_kinematics_gen.py`.

2. **Origin rpy is composed in the wrong order.** `get_transform` builds transposed elementary rotations
   and multiplies them as `yaw @ pitch @ roll`. Read back through the template's column-major convention,
   that gives `R_x(r) R_y(p) R_z(y)`, but URDF is `R_z(y) R_y(p) R_x(r)`. The two only agree when at most
   one angle is non-zero. The reBot's joint6 origin is `rpy="3.14159265 1.57079265 3.14159265"`.
   *Fix:* build `R = R_z R_y R_x` (`rpy_matrix`) and pass `R.T` to the template, because the template
   reads each written 3x3 as R11 R21 R31 R12 ... (column major).

3. **Fixed joints are multiplied element-wise and in the wrong order.** It uses
   `T_fixed = T_fixed * get_transform(joint)` (numpy `*` is element-wise), and the result is applied as
   `get_transform(joint) @ T_fixed`, after the next joint's origin. It should be a matrix product, applied
   before that origin: `T_fixed @ origin`. Any fixed joint between two movable joints gives wrong poses.
   *Fix:* accumulate `pending = pending @ origin` and use `pending @ origin_i @ A_i`.

4. **Fixed joints after the last movable joint are dropped.** The chain walk includes them, but the
   template reports the last joint's frame as the end effector, so a tool frame like the reBot's
   `gripper_end`, 0.166 m past joint6, is lost.
   *Fix:* return what's left after the last joint as a constant tip transform. FK multiplies it on, and
   IK targets `target * tip^-1`. Here it's emitted in the generated chain header and applied in
   `robot_model_utils/fast_kinematics_impl.hpp`. Upstream, the template could apply it to a final frame.

5. **Joint limits are a TODO.** IK can return solutions outside the URDF limits.
   *Fix, applied here as a patch:* `joint_limit_box_constraints.patch` switches `InverseKinematics` to
   LBFGSpp's box-constrained `LBFGSBSolver`, with bounds set by `set_joint_limits(lower, upper)` on
   `InverseKinematics` and `JointData` (unbounded by default). `fast_kinematics_impl.hpp` passes the MuJoCo
   model's limits. Upstream could also emit the URDF limits from the generator.
   Cost: L-BFGS-B is slower per iteration. reBot IK went from ~60 µs to ~330 µs per random target
   (~19 µs to ~100 µs per local solve), with the same 200/200 success.

## Template and headers

6. **Prismatic joints after the first use the previous frame's z.** The prismatic branch uses
   `R13_old/R23_old/R33_old * linear`, the previous frame's z axis. It should use the new frame's,
   `R_old * R_fixed` column 3, like the first-joint branch does with `R13_fixed`. Separately,
   `JointData::set_joint(s)` and the IK cost always write `sin/cos(value)`, which is wrong for prismatic
   joints. The local generator rejects prismatic joints until both are fixed.

7. **IK stops too early.** `InverseKinematics` sets `param.epsilon = epsilon_rel = 1e-3` with
   `max_iterations = 30`. With this cost (squared distances of three points 1 m along the axes), a
   gradient norm of 1e-3 is reached at around millimeter-level error. Only 52% of random reBot targets
   reached a 1 mm / 0.01 rad tolerance (105/200, 20 random restarts). With epsilon 1e-6 and 100
   iterations it was 200/200, at about 3 attempts and 60 µs per target.
   *Fix:* lower the defaults, or make them settable. They are settable already, by luck: `LBFGSSolver`
   keeps a reference to `param`, so changing `fun->param` after construction works.

8. **`-Ofast` breaks the IK line search.** `generate_fast_forward_kinematics_library` adds `-Ofast`
   (`-ffast-math`), but LBFGSpp's line search compares against infinity, which `-ffinite-math-only` makes
   undefined (clang warns `-Wnan-infinity-disabled`). Use `-O3` (plus `-fno-math-errno`) for the IK code;
   FK alone could keep `-Ofast`. `-march=native` also makes builds machine-specific, which defeats shared
   build caches.

9. **One robot per binary.** `FAST_FK_NUMBER_OF_JOINTS` is a global macro that sets the size of
   `JointData`/`InverseKinematics`, and `fast_fk::internal::forward_kinematics_internal` is a global
   symbol. Two generated robots in one program break the one-definition rule and fail to link.
   *Fix:* generate a per-robot namespace and constant. Here the generated chain header does
   `#define fast_fk <robot>_fast_fk` and the backend class sits in an anonymous namespace.

10. **Small header issues:**
    - `fast_inverse_kinematics.hpp`: `int niter;` is uninitialized, but is returned when the solver throws
      (initialized in the patch).
    - `kinematics_interface.hpp` uses `std::string` without `#include <string>`.
    - `JointData`'s constructor initializes `target_pose` before `target_rot`, the reverse of their
      declaration order (`-Wreorder-ctor`).
    - `= { 0 }` on the nested `std::array`s triggers `-Wmissing-braces`.
    - The template has unused `tmp` variables (`-Wunused-but-set-variable`). Some of these warnings come
      from LBFGSpp itself.
