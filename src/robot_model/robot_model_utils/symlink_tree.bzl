"""symlink_tree: lay files out under their MJCF-relative paths, as symlinks (no copies).

MJCFs find meshes by a path relative to the MJCF (<compiler meshdir>), and the meshes of
a composed model come from several repositories. This rule links them into one tree next
to the generated MJCF:

    symlink_tree(
        name = "tree",
        srcs = ["@rebot_arm_description//:rs_meshes", "@orbbec_description//:meshes"],
        # a source whose path contains KEY goes to VALUE + what follows KEY; others are left out
        layout = {"/RS/meshes/": "meshes/", "/meshes/gemini305/": "meshes/gemini305/"},
    )

Outputs are in the package of the rule; the first matching KEY (in `layout` order) wins.
Each output is also in an output group named by its path in the package, so one file of the
tree can be a target of its own (symlink outputs are not addressable by label):

    filegroup(name = "rs_arm_mjcf", srcs = [":tree"], output_group = "rs/rs_arm.xml", data = [":tree"])
"""

def _symlink_tree_impl(ctx):
    outs = []
    groups = {}
    for f in ctx.files.srcs:
        path = "/" + f.short_path
        for key, dest in ctx.attr.layout.items():
            i = path.find(key)
            if i >= 0:
                out = ctx.actions.declare_file(dest + path[i + len(key):])
                ctx.actions.symlink(output = out, target_file = f)
                outs.append(out)
                groups[dest + path[i + len(key):]] = depset([out])
                break
    return [DefaultInfo(files = depset(outs), runfiles = ctx.runfiles(files = outs)), OutputGroupInfo(**groups)]

symlink_tree = rule(
    implementation = _symlink_tree_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "layout": attr.string_dict(mandatory = True),
    },
)
