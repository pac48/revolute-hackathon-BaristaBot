# Local Bazel registry

Bazel modules that aren't in the RCR or the BCR, published the way the RCR does
it: the upstream source archive plus overlaid `BUILD.bazel` files (and patches).
`.bazelrc` lists this registry before the RCR and the BCR.

| Module | Upstream |
|---|---|
| `orbbec_camera_msgs`, `orbbec_description`, `orbbec_camera` | [OrbbecSDK_ROS2](https://github.com/orbbec/OrbbecSDK_ROS2) v2-main @ `ffba347` (2.10.7; first with Gemini 305) |

All three fetch the same tarball (downloaded once); each `strip_prefix`es to its
own package directory. `orbbec_camera` links the prebuilt Orbbec SDK shipped in
`SDK/<arch>/` and installs the SDK's extensions as `lib/extensions/` of its ament
prefix, where the node looks for them.

Layout: `modules/<name>/<version>/{MODULE.bazel, source.json, overlay/, patches/}`.

After editing an overlay or patch:

1. `third_party/registry/update_sources.py` (rewrites the integrity hashes in `source.json`).
2. Build once with `--lockfile_mode=refresh`. Bazel doesn't watch local registry
   files, so without it a running Bazel server (here or on the `.remote` host,
   e.g. `bazel run --lockfile_mode=refresh //src/camera_drivers:orbbec_camera.remote`)
   keeps using the old module.

To move to a newer upstream commit, change `COMMIT` and the archive `integrity` in
`update_sources.py` and bump the module versions.

USB access to Orbbec cameras needs upstream's udev rules on the machine with the
camera (see `src/camera_drivers/orbbec_camera.py`).
