"""Collision-sphere spec for spherizing a URDF for VAMP (cricket).

A spec is a YAML file (checked in, hand-editable; seed_spheres.py writes a
first version):

    # Units: meters. Sphere centers are in each link's URDF frame.
    fixed_joints:          # joints frozen at a position in the planning model
      joint7: 0.0
    spheres:
      link1:
        - [0.0, 0.0, 0.05, 0.04]   # x, y, z, radius
    disable_collisions:    # extra SRDF pairs; rigid/adjacent pairs are automatic
      - [link3, link5]

This module loads specs and does the URDF-side work: freezing joints, replacing
collision geometry with the spheres, computing which link pairs never need
checking, and forward kinematics for the spheres (for checks).
"""

import dataclasses
import itertools
import math
import xml.etree.ElementTree as ET

import numpy as np
import yaml

MOVABLE = ("revolute", "continuous", "prismatic")


@dataclasses.dataclass
class Spec:
    spheres: dict[str, list[tuple[float, float, float, float]]]
    fixed_joints: dict[str, float]
    disable_collisions: list[tuple[str, str]]


def load_spec(path: str) -> Spec:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"spheres", "fixed_joints", "disable_collisions"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    spheres = {}
    for link, items in (raw.get("spheres") or {}).items():
        spheres[link] = []
        for s in items or []:
            if len(s) != 4 or s[3] <= 0:
                raise ValueError(f"{path}: {link}: sphere {s} is not [x, y, z, radius > 0]")
            spheres[link].append(tuple(float(v) for v in s))
    return Spec(
        spheres=spheres,
        fixed_joints={k: float(v) for k, v in (raw.get("fixed_joints") or {}).items()},
        disable_collisions=[tuple(p) for p in raw.get("disable_collisions") or []],
    )


def merge_specs(specs: list[Spec]) -> Spec:
    """One spec from several, for a model composed of parts: each part's spec has its own
    links (a link in two specs is an error); fixed joints and disabled pairs are the union."""
    spheres, fixed, disabled = {}, {}, []
    for spec in specs:
        for link, items in spec.spheres.items():
            if link in spheres:
                raise ValueError(f"link {link!r} has spheres in more than one spec")
            spheres[link] = items
        for joint, value in spec.fixed_joints.items():
            if fixed.setdefault(joint, value) != value:
                raise ValueError(f"joint {joint!r} is fixed at different positions")
        disabled += [p for p in spec.disable_collisions if p not in disabled]
    return Spec(spheres, fixed, disabled)


def _radius_up(r: float) -> float:
    """r rounded up to whole micrometers, plus enough to cover rounding the center
    to micrometers (<= 0.87 um), so a written sphere still covers what it did."""
    return math.ceil(r * 1e6 + 1) / 1e6


def dump_spec(spec: Spec, header: str = "") -> str:
    lines = [f"# {line}".rstrip() for line in header.splitlines()]
    lines.append("# Units: meters. Sphere centers ([x, y, z, radius]) are in each link's URDF frame.")
    lines.append("fixed_joints:" + ("" if spec.fixed_joints else " {}"))
    lines += [f"  {j}: {v:g}" for j, v in spec.fixed_joints.items()]
    lines.append("spheres:" + ("" if spec.spheres else " {}"))
    for link, spheres in spec.spheres.items():
        lines.append(f"  {link}:")
        lines += [f"    - [{x:.6f}, {y:.6f}, {z:.6f}, {_radius_up(r):.6f}]" for x, y, z, r in spheres]
    lines.append("disable_collisions:" + ("" if spec.disable_collisions else " []"))
    lines += [f"  - [{a}, {b}]" for a, b in spec.disable_collisions]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Transforms


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def matrix_rpy(m: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -m[2, 0])))
    if abs(math.cos(pitch)) > 1e-9:
        return math.atan2(m[2, 1], m[2, 2]), pitch, math.atan2(m[1, 0], m[0, 0])
    return math.atan2(-m[1, 2], m[1, 1]), pitch, 0.0


def axis_angle_matrix(axis, angle) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k


def _origin(joint) -> np.ndarray:
    t = np.eye(4)
    o = joint.find("origin")
    if o is not None:
        t[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        t[:3, :3] = rpy_matrix([float(v) for v in o.get("rpy", "0 0 0").split()])
    return t


def _axis(joint) -> np.ndarray:
    a = joint.find("axis")
    return np.array([float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()])


def joint_motion(joint, q: float) -> np.ndarray:
    """Transform of the joint's child frame relative to its origin frame at position q."""
    t = np.eye(4)
    if joint.get("type") in ("revolute", "continuous"):
        t[:3, :3] = axis_angle_matrix(_axis(joint), q)
    elif joint.get("type") == "prismatic":
        t[:3, 3] = _axis(joint) * q
    return t


def _fmt(values) -> str:
    return " ".join(f"{v:.9g}" if abs(v) > 1e-12 else "0" for v in values)


# ---------------------------------------------------------------------------
# URDF processing


def freeze_joints(robot: ET.Element, fixed: dict[str, float]):
    """Make the given joints fixed at their positions (origin absorbs the motion)."""
    joints = {j.get("name"): j for j in robot.iter("joint")}
    missing = set(fixed) - set(joints)
    if missing:
        raise ValueError(f"fixed_joints not in the URDF: {sorted(missing)}")
    for name, q in fixed.items():
        joint = joints[name]
        if joint.get("type") not in MOVABLE:
            continue
        t = _origin(joint) @ joint_motion(joint, q)
        origin = joint.find("origin")
        if origin is None:
            origin = ET.SubElement(joint, "origin")
        origin.set("xyz", _fmt(t[:3, 3]))
        origin.set("rpy", _fmt(matrix_rpy(t[:3, :3])))
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "dynamics", "mimic", "safety_controller", "calibration"):
            for e in joint.findall(tag):
                joint.remove(e)


def set_sphere_collisions(robot: ET.Element, spheres: dict[str, list]):
    """Replace every link's collision geometry with the spec's spheres."""
    links = {l.get("name"): l for l in robot.iter("link")}
    missing = set(spheres) - set(links)
    if missing:
        raise ValueError(f"spheres for links not in the URDF: {sorted(missing)}")
    for name, link in links.items():
        for c in link.findall("collision"):
            link.remove(c)
        for i, (x, y, z, r) in enumerate(spheres.get(name, [])):
            c = ET.SubElement(link, "collision", name=f"{name}_sphere_{i}")
            ET.SubElement(c, "origin", xyz=_fmt((x, y, z)), rpy="0 0 0")
            g = ET.SubElement(c, "geometry")
            ET.SubElement(g, "sphere", radius=f"{r:.9g}")


def rigid_groups(robot: ET.Element) -> dict[str, int]:
    """Link -> group id, where links joined by fixed joints share a group."""
    parent = {l.get("name"): l.get("name") for l in robot.iter("link")}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for j in robot.iter("joint"):
        if j.get("type") not in MOVABLE:
            a, b = find(j.find("parent").get("link")), find(j.find("child").get("link"))
            parent[a] = b
    roots = {}
    return {l: roots.setdefault(find(l), len(roots)) for l in parent}


def automatic_disabled_pairs(robot: ET.Element) -> list[tuple[str, str, str]]:
    """(link, link, reason) for pairs that never need checking: same rigid body or adjacent."""
    group = rigid_groups(robot)
    adjacent_groups = set()
    for j in robot.iter("joint"):
        if j.get("type") in MOVABLE:
            g = (group[j.find("parent").get("link")], group[j.find("child").get("link")])
            adjacent_groups.add(frozenset(g))
    pairs = []
    for a, b in itertools.combinations(sorted(group), 2):
        if group[a] == group[b]:
            pairs.append((a, b, "Rigid"))
        elif frozenset((group[a], group[b])) in adjacent_groups:
            pairs.append((a, b, "Adjacent"))
    return pairs


def srdf(robot: ET.Element, extra_pairs) -> str:
    pairs = automatic_disabled_pairs(robot)
    known = {frozenset(p[:2]) for p in pairs}
    links = {l.get("name") for l in robot.iter("link")}
    for a, b in extra_pairs:
        if a not in links or b not in links:
            raise ValueError(f"disable_collisions pair [{a}, {b}] has a link not in the URDF")
        if frozenset((a, b)) not in known:
            pairs.append((a, b, "User"))
            known.add(frozenset((a, b)))
    root = ET.Element("robot", name=robot.get("name"))
    for a, b, reason in pairs:
        ET.SubElement(root, "disable_collisions", link1=a, link2=b, reason=reason)
    ET.indent(root)
    return '<?xml version="1.0"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def link_poses(robot: ET.Element, q: dict[str, float]) -> dict[str, np.ndarray]:
    """4x4 world pose of every link at joint positions q (missing joints at 0)."""
    joints_by_child = {j.find("child").get("link"): j for j in robot.iter("joint")}
    poses = {}

    def pose(link):
        if link not in poses:
            joint = joints_by_child.get(link)
            if joint is None:
                poses[link] = np.eye(4)
            else:
                poses[link] = (
                    pose(joint.find("parent").get("link"))
                    @ _origin(joint)
                    @ joint_motion(joint, q.get(joint.get("name"), 0.0))
                )
        return poses[link]

    for link in robot.iter("link"):
        pose(link.get("name"))
    return poses


def sphere_positions(robot: ET.Element, spheres: dict[str, list], q: dict[str, float]):
    """World positions of every sphere at joint positions q (missing joints at 0):
    list of (link, center xyz, radius)."""
    poses = link_poses(robot, q)
    out = []
    for link, items in spheres.items():
        for x, y, z, r in items:
            out.append((link, (poses[link] @ np.array([x, y, z, 1.0]))[:3], r))
    return out


def colliding_pairs(robot: ET.Element, spheres, q, disabled) -> dict[frozenset, float]:
    """Link pairs (not disabled) with overlapping spheres at q -> deepest penetration."""
    disabled = {frozenset(p[:2]) for p in disabled}
    placed = sphere_positions(robot, spheres, q)
    out = {}
    for (la, ca, ra), (lb, cb, rb) in itertools.combinations(placed, 2):
        pair = frozenset((la, lb))
        if la == lb or pair in disabled:
            continue
        depth = ra + rb - float(np.linalg.norm(ca - cb))
        if depth > 0:
            out[pair] = max(out.get(pair, 0.0), depth)
    return out
