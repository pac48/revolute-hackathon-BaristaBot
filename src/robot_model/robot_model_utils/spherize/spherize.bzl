"""spherized_robot: URDF + sphere spec -> spherized URDF and SRDF for vamp_robot; sphere_editor: the spheres in Blender."""

load("@rules_python//python:defs.bzl", "py_binary")
load("//src/robot_model/robot_model_utils:viewer.bzl", "model_viewer")

def spherized_robot(name, urdf, spec = None, specs = [], check_config = [], visibility = None):
    """Run spherize_urdf: outputs NAME.urdf and NAME.srdf.

    Fails the build if spheres of two links overlap at the check configuration
    (default: all joints 0) without being disabled; see spherize_urdf.py.

    Args:
      name: target name; the outputs are NAME.urdf and NAME.srdf.
      urdf: source URDF (joint and link names as in the spec).
      spec: sphere spec YAML (spheres_spec.py).
      specs: several specs, merged: for a composed model, the parts' specs (each link in one).
      check_config: ["JOINT=POSITION", ...] for the self-collision check.
      visibility: of the outputs.
    """
    specs = ([spec] if spec else []) + specs
    if not specs:
        fail("{}: give spec or specs".format(name))
    native.genrule(
        name = name,
        srcs = [urdf] + specs,
        outs = [name + ".urdf", name + ".srdf"],
        tools = ["//src/robot_model/robot_model_utils/spherize:spherize_urdf"],
        cmd = ("$(execpath //src/robot_model/robot_model_utils/spherize:spherize_urdf)" +
               " --urdf $(location " + urdf + ") --spec " + " ".join(["$(location " + s + ")" for s in specs]) +
               " --out-urdf $(location " + name + ".urdf) --out-srdf $(location " + name + ".srdf)" +
               (" --check-config " + " ".join(check_config) if check_config else "")),
        visibility = visibility,
    )

def sphere_editor(urdf, specs, mjcf = None, prefixes = [], obj = "spheres.obj"):
    """The Blender round trip for a model's spheres: targets export_spheres and import_spheres, no arguments.

        bazel run //<package>:export_spheres   # writes <package>/spheres.obj: the spheres (and the MJCF's
                                               # geometry) at joint positions 0 and the spec's fixed joints
        (edit the spheres in Blender, export the OBJ over it: Forward Y, Up Z; see import_spheres_obj.py)
        bazel run //<package>:import_spheres   # writes the changes back into the spec files

    Args:
      urdf: the URDF the specs are for (the one given to spherized_robot).
      specs: the spec files (source files of this or other packages) that spherized_robot merges. With
        prefixes, those of one robot.
      mjcf: the model's MJCF, for its geometry in the OBJ (optional).
      prefixes: the name prefixes of the copies of the specs' robot in the URDF (e.g. ["left_", "right_"]).
      obj: the OBJ, relative to this package's directory in the source tree (gitignored).
    """
    pkg = native.package_name()
    obj_path = pkg + "/" + obj
    spec_paths = []
    for spec in specs:
        label = native.package_relative_label(spec)
        spec_paths.append(label.package + "/" + label.name)
    prefix_args = (["--prefix"] + prefixes) if prefixes else []
    data = [urdf] + ([mjcf] if mjcf else [])
    for tool, extra in [
        ("export", ["--out", obj_path] + (["--mjcf", "$(rootpath " + mjcf + ")"] if mjcf else [])),
        ("import", ["--obj", obj_path] + (["--mjcf", "$(rootpath " + mjcf + ")"] if mjcf else [])),
    ]:
        py_binary(
            name = tool + "_spheres",
            srcs = ["//src/robot_model/robot_model_utils/spherize:%s_spheres_obj.py" % tool],
            main = "//src/robot_model/robot_model_utils/spherize:%s_spheres_obj.py" % tool,
            args = ["--urdf", "$(rootpath " + urdf + ")", "--spec"] + spec_paths + prefix_args + extra,
            data = data,
            deps = ["//src/robot_model/robot_model_utils/spherize:sphere_editor_lib"],
        )

def _spheres_mjcf_impl(ctx):
    pkg = ctx.attr.tree.label.package
    specs = ctx.files.specs
    outs = []
    groups = {}
    for f in ctx.files.tree:
        rel = f.short_path[len(pkg) + 1:] if pkg else f.short_path
        out = ctx.actions.declare_file(ctx.label.name + "/" + rel)
        if f.basename.endswith(".xml"):
            args = ctx.actions.args()
            args.add("--spec").add_all(specs)
            args.add("--in", f)
            args.add("--out", out)
            ctx.actions.run(
                executable = ctx.executable._tool,
                arguments = [args],
                inputs = [f] + specs,
                outputs = [out],
                mnemonic = "SpheresMjcf",
                progress_message = "Spherizing %s" % rel,
            )
        else:
            ctx.actions.symlink(output = out, target_file = f)  # meshes etc., next to the rewritten XML
        outs.append(out)
        groups[rel] = depset([out])
    return [DefaultInfo(files = depset(outs), runfiles = ctx.runfiles(files = outs)), OutputGroupInfo(**groups)]

_spheres_mjcf = rule(
    implementation = _spheres_mjcf_impl,
    attrs = {
        "tree": attr.label(mandatory = True),
        "specs": attr.label_list(allow_files = True, mandatory = True),
        "_tool": attr.label(
            default = "//src/robot_model/robot_model_utils/spherize:spheres_mjcf",
            executable = True,
            cfg = "exec",
        ),
    },
)

def spheres_model(tree, scene, specs, visibility = None):
    """The model's MJCF with the spec's spheres as its collision geoms, and a viewer for it.

    Targets (in the calling package):
      spheres_files: the data: every file of `tree` under spheres_files/ (XML rewritten, meshes linked)
      spheres_scene.xml: `scene` of it (the MJCF)
      view_spheres: bazel run :view_spheres opens it in MuJoCo's viewer (the spheres are geom group 2)

    Args:
      tree: the target of the model's MJCF files and meshes (a symlink_tree), as the model's viewer uses.
      scene: the model's main MJCF in `tree`, e.g. "scene.xml" (output-group path).
      specs: the sphere spec files (as for spherized_robot), names as in the MJCF's bodies.
      visibility: of the data.
    """
    _spheres_mjcf(name = "spheres_files", tree = tree, specs = specs, visibility = visibility)
    native.filegroup(
        name = "spheres_scene.xml",
        srcs = [":spheres_files"],
        output_group = scene,
        data = [":spheres_files"],
        visibility = visibility,
    )
    model_viewer(name = "view_spheres", mjcf = ":spheres_scene.xml")
