"""Spherize a URDF from a sphere spec: write the URDF and SRDF cricket needs.

    spherize_urdf --urdf robot.urdf --spec spheres.yaml --out-urdf out.urdf --out-srdf out.srdf

The output URDF has the spec's fixed_joints frozen and every link's collision
geometry replaced by the spec's spheres. The SRDF disables links that are
rigidly connected or adjacent (joined by one movable joint), plus the spec's
disable_collisions. The output fails if, at the home configuration (all
remaining joints at 0, or --check-config), spheres of two links overlap without
being disabled: the planner would see the robot colliding with itself.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

from src.robot_model.robot_model_utils.spherize import spheres_spec


def parse_config(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        joint, _, value = item.partition("=")
        out[joint] = float(value)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--spec", required=True, nargs="+",
                    help="sphere spec(s); several are merged (spheres_spec.merge_specs)")
    ap.add_argument("--out-urdf", required=True)
    ap.add_argument("--out-srdf", required=True)
    ap.add_argument("--check-config", nargs="*", default=[], metavar="JOINT=POSITION",
                    help="configuration for the self-collision check (default: all joints 0)")
    ap.add_argument("--no-check", action="store_true", help="skip the self-collision check")
    args = ap.parse_args()

    spec = spheres_spec.merge_specs([spheres_spec.load_spec(p) for p in args.spec])
    tree = ET.parse(args.urdf)
    robot = tree.getroot()
    spheres_spec.freeze_joints(robot, spec.fixed_joints)
    spheres_spec.set_sphere_collisions(robot, spec.spheres)
    srdf = spheres_spec.srdf(robot, spec.disable_collisions)

    if not args.no_check:
        disabled = spheres_spec.automatic_disabled_pairs(robot) + list(spec.disable_collisions)
        hits = spheres_spec.colliding_pairs(robot, spec.spheres, parse_config(args.check_config), disabled)
        if hits:
            print(f"spherize_urdf: {" ".join(args.spec)}: spheres overlap at the check configuration:", file=sys.stderr)
            for pair, depth in sorted(hits.items(), key=lambda kv: -kv[1]):
                a, b = sorted(pair)
                print(f"  [{a}, {b}]  penetration {depth * 1000:.1f} mm", file=sys.stderr)
            print("Shrink/move those spheres, or add the pairs to disable_collisions.", file=sys.stderr)
            return 1

    ET.indent(robot)
    with open(args.out_urdf, "w") as f:
        f.write('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n")
    with open(args.out_srdf, "w") as f:
        f.write(srdf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
