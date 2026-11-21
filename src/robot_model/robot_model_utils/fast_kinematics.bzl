"""fast_kinematics: a robot's robot_model::Kinematics from fast_robot_kinematics code generation.

    fast_kinematics(
        name = "rebot_arm_kinematics",
        urdf = "//src/robot_model/rebot_arm_data:rs_generated.urdf",
        root_link = "base_link",
        tip_link = "gripper_end",
        namespace = "rebot_arm_kinematics",
    )

gives a cc_library `name` with header `<package>/<name>.hpp` declaring

    std::unique_ptr<robot_model::Kinematics> <namespace>::make_kinematics(const mjModel* model);

The FK and IK cost/gradient are generated from the URDF at build time
(robot_model_utils/fast_kinematics_gen.py rendering fast_robot_kinematics'
template). The URDF's joint names must match the MuJoCo model's. Each robot's
generated code lives in its own namespace, so several can link into one binary.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

# The generated FK/IK is the hot loop: optimize it regardless of compilation mode.
# Not -Ofast/-ffast-math as upstream: LBFGSpp's line search relies on infinity,
# which -ffinite-math-only makes undefined.
_FK_COPTS = [
    "-O3",
    "-fno-math-errno",
    # Upstream code (fast_robot_kinematics headers, LBFGSpp).
    "-Wno-missing-braces",
    "-Wno-unused-but-set-variable",
    "-Wno-shadow",
    "-Wno-reorder-ctor",
]

def fast_kinematics(name, urdf, root_link, tip_link, namespace, visibility = None, **kwargs):
    """Generates the code and the cc_library `name`.

    Args:
      name: cc_library name (also the generated files' prefix).
      urdf: URDF label.
      root_link: chain root; poses are in its frame.
      tip_link: chain tip; the link whose pose FK returns and IK targets.
      namespace: C++ namespace of make_kinematics (a valid identifier).
      visibility: of the cc_library.
      **kwargs: passed to the cc_library (e.g. tags).
    """
    package = native.package_name()
    outs = [name + suffix for suffix in ("_fk.cpp", "_chain.hpp", ".hpp", ".cpp")]
    native.genrule(
        name = name + "_gen",
        srcs = [urdf, "@fast_robot_kinematics//:code_generation/robot_config.cpp.template"],
        outs = outs,
        cmd = " ".join([
            "$(execpath //src/robot_model/robot_model_utils:fast_kinematics_gen)",
            "$(execpath {})".format(urdf),
            "$(execpath @fast_robot_kinematics//:code_generation/robot_config.cpp.template)",
            root_link,
            tip_link,
            namespace,
            package,
            name,
            "$(RULEDIR)",
        ]),
        tools = ["//src/robot_model/robot_model_utils:fast_kinematics_gen"],
        visibility = ["//visibility:private"],
    )
    deps = [
        "//src/robot_model/robot_model_utils:fast_kinematics_impl",
        "//src/robot_model/robot_model_utils:kinematics",
        "@fast_robot_kinematics//:headers",
    ]
    cc_library(
        name = name + "_fk",
        srcs = [name + "_fk.cpp"],
        hdrs = [name + "_chain.hpp"],
        copts = _FK_COPTS,
        deps = deps,
        visibility = ["//visibility:private"],
    )
    cc_library(
        name = name,
        srcs = [name + ".cpp"],
        copts = _FK_COPTS,
        hdrs = [name + ".hpp"],
        deps = deps + [":" + name + "_fk"],
        visibility = visibility,
        **kwargs
    )
