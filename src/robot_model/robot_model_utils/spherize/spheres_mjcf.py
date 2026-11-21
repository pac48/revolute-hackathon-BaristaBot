"""Rewrite an MJCF file so each spherized body collides as its spheres.

    spheres_mjcf --spec spheres.yaml [more.yaml ...] --in model.xml --out model_spheres.xml

For every <body> whose name is a link of the spec (names as in this file: an attached model's bodies are
unprefixed here, so its copies share the spheres, as they share the spec), the body's collision geoms
(contype/conaffinity not 0) are replaced by one sphere geom per sphere of the spec; visual geoms are kept.
The spheres are translucent orange in geom group 2 (shown by default in MuJoCo's viewer). Other bodies and
files without spherized bodies are unchanged (comments are dropped). Run per file by spheres_mjcf in spherize.bzl,
which links the rest of the model (meshes) next to the outputs.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

from src.robot_model.robot_model_utils.spherize import mjcf_collision as mc
from src.robot_model.robot_model_utils.spherize import spheres_spec

RGBA = "1 0.45 0.1 0.4"


def spherize_tree(root: ET.Element, spheres: dict[str, list]) -> set[str]:
    """Replaces the collision geoms of the bodies in `spheres` in place; returns the bodies done."""
    done = set()
    for body in root.iter("body"):
        name = body.get("name")
        if name not in spheres:
            continue
        for geom in [g for g in body.findall("geom") if mc.is_collision(g)]:
            body.remove(geom)
        for x, y, z, r in spheres[name]:
            ET.SubElement(body, "geom", {"type": "sphere", "pos": f"{x:.6f} {y:.6f} {z:.6f}", "size": f"{r:.6f}",
                                         "rgba": RGBA, "group": "2", "contype": "1", "conaffinity": "1"})
        done.add(name)
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", nargs="+", required=True)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = spheres_spec.merge_specs([spheres_spec.load_spec(p) for p in args.spec])
    tree = ET.parse(args.src)
    spherize_tree(tree.getroot(), spec.spheres)
    ET.indent(tree)
    tree.write(args.out, encoding="unicode")
    return 0


if __name__ == "__main__":
    sys.exit(main())
