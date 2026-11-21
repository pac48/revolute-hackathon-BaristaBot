"""Generate a robot's fast_robot_kinematics code from its URDF.

    fast_kinematics_gen.py URDF TEMPLATE ROOT_LINK TIP_LINK NAMESPACE PACKAGE NAME OUT_DIR

Writes, for the joints between ROOT_LINK and TIP_LINK (used by fast_kinematics.bzl):
  NAME_fk.cpp     fast_robot_kinematics' robot_config.cpp.template rendered
                  (unrolled FK + the IK cost/gradient)
  NAME_chain.hpp  joint count, joint names and the constant transform from the
                  last joint's frame to the tip; configures upstream's headers
  NAME.hpp        NAMESPACE::make_kinematics(model) -> robot_model::Kinematics
  NAME.cpp        its definition (robot_model_utils/fast_kinematics_impl.hpp)

This replaces upstream's code_generation/robot_gen.py, which assumes every joint
rotates about +z, composes origin rpy as R_x R_y R_z instead of URDF's R_z R_y R_x,
multiplies fixed joints element-wise, and stops at the last joint. Here every
joint's frame is rotated by a constant A_i (A_i z = axis_i), so the template's
z rotation is the joint's rotation: origin_i * Rot(axis_i, q) = origin_i * A_i *
Rot_z(q) * A_i^T; A_i^T and any fixed joints are folded into the next segment and
what's left after the last joint becomes the tip transform. Revolute and
continuous joints only (the template's prismatic code assumes a z-aligned
previous frame, and its joint setters write sin/cos).
"""

import argparse
import math
import xml.etree.ElementTree as ET

import jinja2
import numpy as np


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF origin rotation: R_z(yaw) R_y(pitch) R_x(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw),
                              math.sin(yaw))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def align_z(axis) -> np.ndarray:
    """A rotation A with A @ z = axis (normalized)."""
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ a)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])  # rotation by pi about x
    v = np.cross(z, a)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def homogeneous(rotation: np.ndarray, translation=(0.0, 0.0, 0.0)) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def chain(urdf_text: str, root: str, tip: str) -> list[dict]:
    """Joints from root to tip: {name, type, origin (4x4), axis}."""
    by_child = {}
    for j in ET.fromstring(urdf_text).iter("joint"):
        origin = j.find("origin")
        xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0.0] * 3
        rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0.0] * 3
        axis = j.find("axis")
        by_child[j.find("child").get("link")] = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "origin": homogeneous(rpy_matrix(*rpy), xyz),
            "axis": [float(v) for v in axis.get("xyz").split()] if axis is not None else [1.0, 0.0, 0.0],
        }
    joints, link = [], tip
    while link != root:
        if link not in by_child:
            raise ValueError(f"{tip!r} is not below {root!r} in the URDF")
        joints.append(by_child[link])
        link = by_child[link]["parent"]
    return joints[::-1]


def segments(joints: list[dict]) -> tuple[list[dict], np.ndarray]:
    """Template segments (name, rotation, offset) and the last joint frame -> tip transform."""
    out, pending = [], np.eye(4)
    for j in joints:
        if j["type"] == "fixed":
            pending = pending @ j["origin"]
        elif j["type"] in ("revolute", "continuous"):
            a = homogeneous(align_z(j["axis"]))
            m = pending @ j["origin"] @ a
            out.append({"name": j["name"], "rotation": m[:3, :3], "offset": m[:3, 3]})
            pending = a.T
        else:
            raise ValueError(f"joint {j['name']!r} is {j['type']}; only revolute/continuous joints are supported")
    if not out:
        raise ValueError("no movable joints between root and tip")
    return out, pending


def clean(values: np.ndarray) -> list:
    """Exact zeros for round-off (e.g. cos(pi/2)), so the compiler can drop the terms."""
    return np.where(np.abs(values) < 1e-12, 0.0, values).tolist()


def render(template_text: str, segs: list[dict]) -> str:
    # The template reads each written 3x3 as R11 R21 R31 R12 ... (column major).
    return jinja2.Template(template_text, trim_blocks=True).render(
        rotations=[clean(s["rotation"].T) for s in segs],
        offsets=[clean(s["offset"]) for s in segs],
        types=["revolute"] * len(segs),
    )


def chain_header(namespace: str, root: str, tip: str, segs: list[dict], tip_transform: np.ndarray) -> str:
    names = ", ".join(f'"{s["name"]}"' for s in segs)
    tip_values = ", ".join(repr(float(v)) for row in clean(tip_transform) for v in row)
    return f"""// Generated by fast_kinematics_gen.py: the {root} -> {tip} chain.
// Private to the fast_kinematics() library: it configures fast_robot_kinematics'
// headers (joint count, IK on) and gives their fast_fk namespace a per-robot name.
#pragma once

#include <array>

#define FAST_FK_NUMBER_OF_JOINTS {len(segs)}
#define FAST_FK_USE_IK
#define fast_fk {namespace}_fast_fk

namespace {namespace}::chain
{{
inline constexpr const char* kRootLink = "{root}";
inline constexpr const char* kTipLink = "{tip}";
inline constexpr std::array<const char*, {len(segs)}> kJointNames = {{{names}}};
// Last joint's (axis-aligned) frame -> tip, row major.
inline constexpr std::array<double, 16> kTipTransform = {{{tip_values}}};
}}  // namespace {namespace}::chain
"""


def api_header(namespace: str, root: str, tip: str) -> str:
    return f"""// Generated by fast_kinematics_gen.py.
#pragma once

#include "src/robot_model/robot_model_utils/kinematics.hpp"

#include <memory>

namespace {namespace}
{{
// Kinematics of the {root} -> {tip} chain of `model` (not owned; must outlive it).
std::unique_ptr<robot_model::Kinematics> make_kinematics(const mjModel* model);
}}  // namespace {namespace}
"""


def glue(namespace: str, include_dir: str, name: str) -> str:
    return f"""// Generated by fast_kinematics_gen.py.
#include "{include_dir}/{name}_chain.hpp"  // first: configures fast_robot_kinematics' headers
#include "{include_dir}/{name}.hpp"
#include "src/robot_model/robot_model_utils/fast_kinematics_impl.hpp"

namespace {namespace}
{{
std::unique_ptr<robot_model::Kinematics> make_kinematics(const mjModel* model)
{{
  return std::make_unique<FastKinematics>(model, chain::kJointNames, chain::kRootLink, chain::kTipLink,
                                          chain::kTipTransform);
}}
}}  // namespace {namespace}
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for arg in ("urdf", "template", "root_link", "tip_link", "namespace", "package", "name", "out_dir"):
        ap.add_argument(arg)
    args = ap.parse_args()
    with open(args.urdf) as f:
        segs, tip_transform = segments(chain(f.read(), args.root_link, args.tip_link))
    with open(args.template) as f:
        code = render(f.read(), segs)
    files = {
        f"{args.name}_fk.cpp": f'// Generated by fast_kinematics_gen.py from fast_robot_kinematics\' template.\n'
                               f'#include "{args.package}/{args.name}_chain.hpp"\n\n{code}',
        f"{args.name}_chain.hpp": chain_header(args.namespace, args.root_link, args.tip_link, segs, tip_transform),
        f"{args.name}.hpp": api_header(args.namespace, args.root_link, args.tip_link),
        f"{args.name}.cpp": glue(args.namespace, args.package, args.name),
    }
    for file_name, content in files.items():
        with open(f"{args.out_dir}/{file_name}", "w") as f:
            f.write(content)


if __name__ == "__main__":
    main()
