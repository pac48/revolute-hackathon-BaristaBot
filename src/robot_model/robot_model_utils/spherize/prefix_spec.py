"""Writes a sphere spec with every link and joint name prefixed: a spec of one robot, for a copy of it in a model.

    prefix_spec.py --prefix left_ --spec rs_spheres.yaml camera_spheres.yaml --out left_spheres.yaml

The reBot's spec (rebot_arm_data:rs_spheres.yaml) and its camera's name the single arm's links
and joints; a model with two arms has them as left_link1, right_joint7, ... The spheres are in each
link's own frame, so they are the same: only the names change. (No re-seeding.)
"""

import argparse

from src.robot_model.robot_model_utils.spherize import spheres_spec


def prefixed(spec: spheres_spec.Spec, prefix: str) -> spheres_spec.Spec:
    return spheres_spec.Spec(
        spheres={prefix + link: items for link, items in spec.spheres.items()},
        fixed_joints={prefix + joint: value for joint, value in spec.fixed_joints.items()},
        disable_collisions=[(prefix + a, prefix + b) for a, b in spec.disable_collisions],
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="specs of the parts of one robot (merged)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = spheres_spec.merge_specs([spheres_spec.load_spec(p) for p in args.spec])
    with open(args.out, "w") as f:
        f.write(spheres_spec.dump_spec(prefixed(spec, args.prefix), f"The single arm's spheres, names prefixed {args.prefix!r} (prefix_spec.py). Do not edit."))


if __name__ == "__main__":
    main()
