"""model_viewer: a `bazel run` target that opens a model's MJCF in MuJoCo's viewer.

    load("//src/robot_model/robot_model_utils:viewer.bzl", "model_viewer")

    model_viewer(name = "view", mjcf = "//src/robot_model/two_link_arm_data:scene.xml")
    # bazel run //src/robot_model/two_link_arm:view

`mjcf` is the model's MJCF; `data` adds what it loads by relative path if the MJCF's own
target doesn't carry it (meshes, included files). See viewer.py.
"""

load("@rules_python//python:defs.bzl", "py_binary")

def model_viewer(name, mjcf, data = [], **kwargs):
    py_binary(
        name = name,
        srcs = [Label("//src/robot_model/robot_model_utils:viewer.py")],
        main = Label("//src/robot_model/robot_model_utils:viewer.py"),
        args = ["$(rlocationpath %s)" % mjcf],
        data = [mjcf] + data,
        deps = [Label("//src/robot_model/robot_model_utils:viewer")],
        # Every viewer shares viewer.py; precompiling it per target would emit the same .pyc
        # from conflicting actions.
        precompile = "disabled",
        **kwargs
    )
