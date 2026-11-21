"""Import collision spheres edited in Blender (an OBJ) back into a sphere spec.

    import_spheres_obj --urdf robot.urdf --spec spheres.yaml [more.yaml ...] --obj edited.obj
        [--config JOINT=POS ...] [--prefix left_ right_]

The inverse of export_spheres_obj (give both the same --urdf and --config; the config the OBJ was
exported at is in its "# config:" line and is the default here). Blender: File > Export > Wavefront
(.obj) with Forward axis Y, Up axis Z (the importer's defaults), scale 1, "Apply modifiers", and
"Selection only" off (or select just what to keep). Each link is laid out apart from the others; a
sphere is placed relative to its link's frame, found from the link's mesh_<link> object if --mjcf is
given (so a link may be moved with its mesh), else from the pose the OBJ's header records. Objects are
read by name:

    sphere*                    one sphere: the center is the mean of the object's vertices and the radius
                               their mean distance from it, so move, scale and duplicate freely and any
                               mesh works. Its link is the one whose ball (recorded in the header: the
                               ball around the link's geometry as laid out) it is in, or nearest to: no
                               naming of the link is needed, just keep a sphere with its link's geometry.
    anything else              (mesh_*, the robot's geometry) ignored.

Only links whose spheres changed are touched: a link with no sphere in its ball ends up with none, so
delete only what is meant to go. The specs that changed are rewritten in place (without their comments;
fixed_joints and disable_collisions are kept). With --prefix, the copies of the robot share the spec, so
a link's copies must be edited alike, or only one of them.
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

from src.robot_model.robot_model_utils.spherize import export_spheres_obj, spheres_spec


def read_obj_objects(path: str) -> dict[str, np.ndarray]:
    """Vertices of each OBJ object ("o" or "g" line), by name. Faces are not needed: only the vertices
    each object's faces use (so a vertex list shared across objects, as Blender writes it, is fine)."""
    vertices, used, names = [], {}, []
    current = None
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] in ("o", "g"):
                current = " ".join(parts[1:])
                used.setdefault(current, set())
            elif parts[0] == "v":
                vertices.append([float(c) for c in parts[1:4]])
            elif parts[0] == "f":
                if current is None:
                    current = "unnamed"
                    used.setdefault(current, set())
                for token in parts[1:]:
                    i = int(token.split("/")[0])
                    used[current].add(i - 1 if i > 0 else len(vertices) + i)
    vertices = np.array(vertices)
    return {name: vertices[sorted(ids)] for name, ids in used.items() if ids}


def _header_lines(path: str, key: str):
    with open(path) as f:
        for line in f:
            if line.startswith(f"# {key} "):
                _, _, link, *values = line.split()
                yield link, [float(v) for v in values]
            elif not line.startswith("#"):
                break


def read_header_poses(path: str) -> dict[str, np.ndarray]:
    """The links' poses in the OBJ, from its "# link_pose <link> <16 numbers>" lines."""
    return {link: np.array(v).reshape(4, 4) for link, v in _header_lines(path, "link_pose")}


def read_header_balls(path: str) -> dict[str, tuple[np.ndarray, float]]:
    """The ball around each link's geometry, (center in the link frame, radius), from "# link_ball" lines."""
    return {link: (np.array(v[:3]), v[3]) for link, v in _header_lines(path, "link_ball")}


def fit_rigid(local: np.ndarray, world: np.ndarray) -> np.ndarray | None:
    """The 4x4 pose that takes points `local` to `world` (Kabsch), None if the fit is poor (the mesh was
    edited, not just moved)."""
    lc, wc = local.mean(axis=0), world.mean(axis=0)
    u, _, vt = np.linalg.svd((local - lc).T @ (world - wc))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1, 1, d]) @ u.T
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = wc - rot @ lc
    err = np.abs(local @ rot.T + pose[:3, 3] - world).max()
    return pose if err < 1e-4 else None


def link_poses_in_obj(objects, header, assembled, link_meshes=None) -> dict[str, np.ndarray]:
    """Where each link's frame is in the OBJ: the pose of its mesh_<link> object if that was only moved
    (the spheres then follow the mesh they were placed against), else the header's pose (the layout the
    file was exported with), else the assembled robot's."""
    poses = dict(assembled)
    poses.update(header)
    for link, local in (link_meshes or {}).items():
        world = objects.get(f"mesh_{link}")
        if world is None or world.shape != local.shape:
            continue
        pose = fit_rigid(local, world)
        if pose is None:
            print(f"import_spheres_obj: mesh_{link} was edited, not just moved; the layout's pose is used", file=sys.stderr)
        else:
            poses[link] = pose
    return poses


def is_sphere_object(name: str) -> bool:
    """Objects whose name starts with "sphere" (any case: Blender's own "Sphere" and its ".001" copies
    included) are spheres; everything else (mesh_*, the robot's geometry) is ignored."""
    return re.sub(r"\.\d+$", "", name).lower().startswith(("sphere", "icosphere"))


def parse_name(name: str, links: set[str]) -> str | None:
    """The link named by a sphere_<link>_<rest> object (the longest known link name the rest starts with),
    None if there is none; used when the OBJ has no link balls."""
    name = re.sub(r"\.\d+$", "", name)
    rest = name[len("sphere_"):] if name.startswith("sphere_") else ""
    matches = [link for link in links if rest == link or rest.startswith(link + "_")]
    return max(matches, key=len) if matches else None


def nearest_link(center: np.ndarray, poses, balls) -> str:
    """The link whose ball the point is in (or nearest to): the smallest distance to a ball's surface."""
    return min(balls, key=lambda link: np.linalg.norm(center - (poses[link] @ np.append(balls[link][0], 1.0))[:3]) - balls[link][1])


def spheres_from_objects(objects: dict[str, np.ndarray], poses: dict[str, np.ndarray], balls=None) -> dict[str, list]:
    """The spheres of the sphere objects (world vertices) in their links' frames, in object order. A sphere
    belongs to the link whose ball (`balls`, placed by `poses`) it is in; without balls, to the link its
    name says (sphere_<link>_...)."""
    out = {}
    for name, verts in objects.items():
        if not is_sphere_object(name):
            continue
        center = verts.mean(axis=0)
        link = nearest_link(center, poses, balls) if balls else parse_name(name, set(poses))
        if link is None:
            raise ValueError(f"object {name!r}: the OBJ has no link balls, and no link of the URDF is named at its start (sphere_<link>_...)")
        radius = float(np.linalg.norm(verts - center, axis=1).mean())
        if radius <= 0:
            raise ValueError(f"object {name!r} has no extent")
        local = np.linalg.inv(poses[link]) @ np.append(center, 1.0)
        out.setdefault(link, []).append((float(local[0]), float(local[1]), float(local[2]), radius))
    return out


def changed(before: list, after: list) -> bool:
    return len(before) != len(after) or not np.allclose(np.array(before).reshape(-1, 4), np.array(after).reshape(-1, 4), atol=5e-6)


def resolve(specs: list[spheres_spec.Spec], prefixes: list[str], new: dict[str, list]) -> dict[str, list]:
    """The new spheres by unprefixed link, for the links that changed. A link of a spec has one set of spheres
    for all the prefixes (the copies of the robot share the spec): the copies that were edited must agree."""
    plain = {}
    for link in new:
        match = [p for p in prefixes if link.startswith(p)]
        if not match:
            raise ValueError(f"link {link!r} starts with none of the prefixes {prefixes}")
        plain.setdefault(link[len(max(match, key=len)):], None)
    known = {link: items for spec in specs for link, items in spec.spheres.items()}
    result = {}
    for link in sorted(set(known) | set(plain)):
        before = known.get(link, [])
        edits = [new.get(p + link, []) for p in prefixes]
        edits = [e for e in edits if changed(before, e)]
        if not edits:
            continue
        if any(changed(edits[0], e) for e in edits[1:]):
            raise ValueError(f"link {link!r}: the copies {prefixes} were edited differently, but share one spec")
        result[link] = edits[0]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--spec", nargs="+", required=True,
                    help="the specs the OBJ was exported from, rewritten in place (a new link goes into the first)")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--prefix", nargs="*", default=[""], help="as for export_spheres_obj")
    ap.add_argument("--mjcf", help="the MJCF given to export_spheres_obj: a link's spheres are then placed relative to "
                                   "its mesh_<link> object, so moving a link's mesh and spheres together is fine")
    ap.add_argument("--config", nargs="*", metavar="JOINT=POS",
                    help="joint positions the OBJ was exported at (default: its '# config:' line)")
    args = ap.parse_args()

    def workspace(path):
        if not os.path.isabs(path) and "BUILD_WORKSPACE_DIRECTORY" in os.environ:
            return os.path.join(os.environ["BUILD_WORKSPACE_DIRECTORY"], path)
        return path

    obj_path = workspace(args.obj)
    if not os.path.exists(obj_path):
        print(f"import_spheres_obj: {obj_path} does not exist (export_spheres writes it)", file=sys.stderr)
        return 1
    config = args.config
    if config is None:
        config = []
        with open(obj_path) as f:
            for line in f:
                if line.startswith("# config:"):
                    config = line.split(":", 1)[1].split()
                    break
    q = {j: float(v) for j, _, v in (c.partition("=") for c in config)}

    paths = [workspace(p) for p in args.spec]
    specs = [spheres_spec.load_spec(p) for p in paths]
    merged = export_spheres_obj.expanded_spec(paths, args.prefix)
    robot = ET.parse(args.urdf).getroot()
    spheres_spec.freeze_joints(robot, merged.fixed_joints)
    poses = spheres_spec.link_poses(robot, q)

    objects = read_obj_objects(obj_path)
    meshes = export_spheres_obj.link_meshes(args.mjcf, poses) if args.mjcf else None
    poses = link_poses_in_obj(objects, read_header_poses(obj_path), poses, meshes)
    new = spheres_from_objects(objects, poses, read_header_balls(obj_path) or None)
    if not new:
        print("import_spheres_obj: no sphere_* objects in the OBJ; nothing written", file=sys.stderr)
        return 1
    edits = resolve(specs, args.prefix, new)
    for link, items in edits.items():
        owner = next((s for s in specs if link in s.spheres), specs[0])
        print(f"  {link}: {len(owner.spheres.get(link, []))} -> {len(items)} spheres", file=sys.stderr)
        owner.spheres[link] = items
    for path, spec in zip(paths, specs):
        if any(link in spec.spheres for link in edits):
            with open(path, "w") as f:
                f.write(spheres_spec.dump_spec(spec, "Collision spheres for VAMP, edited in Blender (import_spheres_obj)."))
            print(f"import_spheres_obj: wrote {path}", file=sys.stderr)
    if not edits:
        print("import_spheres_obj: no sphere changed", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
