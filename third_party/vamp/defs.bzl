"""Compiler flags and code generation for VAMP.

VAMP_COPTS: compiler flags for code that includes VAMP headers.
vamp_robot: generate a VAMP robot (cricket's fkcc_gen) from a spherized URDF.

VAMP's SIMD collision checking needs AVX2 + FMA on x86 (its CMake uses
-march=native -mavx2). Explicit flags instead of -march=native keep builds
identical (and cache-shared) across machines; both the dev PC and the remote
build host support them. VAMP is header-only, so these must be on every target
whose sources include VAMP headers; keep VAMP behind a non-VAMP API so only
those sources need them.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

VAMP_COPTS = select({
    "@platforms//cpu:x86_64": ["-mavx2", "-mfma"],
    "//conditions:default": [],
}) + [
    # VAMP's release math flags (cmake/CompilerSettings.cmake), for vectorization.
    "-fno-math-errno",
    "-fno-signed-zeros",
    "-fno-trapping-math",
    "-ffp-contract=fast",
    "-Wno-ignored-attributes",
]

def vamp_robot(name, struct_name, urdf, srdf, end_effector, resolution = 32, module = None, visibility = None):
    """VAMP robot code generated at build time from a spherized URDF.

    Runs cricket's fkcc_gen (tracing compiler) on `urdf` (collision geometry
    must be spheres only) and `srdf` (disabled self-collision pairs), and
    provides the header as a cc_library at <vamp/robots/MODULE.hh>, defining
    vamp::robots::STRUCT_NAME. Compile users with VAMP_COPTS.

    Args:
      name: cc_library name.
      struct_name: C++ struct name of the robot (a valid identifier).
      urdf: spherized URDF label.
      srdf: SRDF label with the disabled collision pairs.
      end_effector: frame used for attachments.
      resolution: collision checking resolution (VAMP uses 32, 64 for Baxter).
      module: header name (default: name).
      visibility: of the cc_library.
    """
    module = module or name
    header = "%s_include/vamp/robots/%s.hh" % (name, module)
    fk_template = "@cricket//:resources/templates/fk_template.hh"
    ccfk_template = "@cricket//:resources/templates/ccfk_template.hh"
    native.genrule(
        name = name + "_codegen",
        srcs = [urdf, srdf, fk_template, ccfk_template],
        outs = [header],
        tools = ["@cricket//:fkcc_gen"],
        # fkcc_gen resolves paths relative to its config, so give it absolute ones.
        cmd = """
set -euo pipefail
dir=$$(mktemp -d)
cat > $$dir/robot.json <<EOF
{{
  "name": "{struct_name}",
  "urdf": "$$PWD/$(location {urdf})",
  "srdf": "$$PWD/$(location {srdf})",
  "end_effector": "{end_effector}",
  "resolution": {resolution},
  "template": "$$PWD/$(location {fk_template})",
  "subtemplates": [{{"name": "ccfk", "template": "$$PWD/$(location {ccfk_template})"}}],
  "output": "$$PWD/$@"
}}
EOF
if ! $(execpath @cricket//:fkcc_gen) $$dir/robot.json > $$dir/log 2>&1; then cat $$dir/log >&2; exit 1; fi
test -s $@
""".format(
            struct_name = struct_name,
            urdf = urdf,
            srdf = srdf,
            end_effector = end_effector,
            resolution = resolution,
            fk_template = fk_template,
            ccfk_template = ccfk_template,
        ),
    )
    cc_library(
        name = name,
        hdrs = [header],
        strip_include_prefix = name + "_include",
        visibility = visibility,
        deps = ["@vamp//:vamp"],
    )
