#!/usr/bin/env python3
"""Rewrites each module's source.json: archive + overlay integrity hashes.

    third_party/registry/update_sources.py

Every module version here fetches the same OrbbecSDK_ROS2 tarball (ARCHIVE)
and overlays the files under its overlay/ directory, as in the RCR (Bazel adds
the registry's MODULE.bazel itself). Run after editing any overlay or patch.
"""

import base64
import hashlib
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
COMMIT = "ffba347cadc96d0457a91af59fc06cdf86052728"  # v2-main, 2026-11-10 (2.10.7; adds Gemini 305)
ARCHIVE = {
    "url": f"https://github.com/orbbec/OrbbecSDK_ROS2/archive/{COMMIT}.tar.gz",
    "integrity": "sha256-v6CfjlVb+QzF12IkZX+PJpzSEQ/Ya/r6eRvr37A4ADE=",
}


def integrity(path: pathlib.Path) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()


def main():
    for version_dir in sorted(p for p in ROOT.glob("modules/*/*") if p.is_dir()):
        module = version_dir.parent.name
        overlay = version_dir / "overlay"
        source = {
            **ARCHIVE,
            "strip_prefix": f"OrbbecSDK_ROS2-{COMMIT}/{module}",
            "overlay": {
                str(p.relative_to(overlay)): integrity(p)
                for p in sorted(overlay.rglob("*")) if p.is_file()
            },
        }
        patches = sorted((version_dir / "patches").glob("*.patch"))
        if patches:
            source["patch_strip"] = 1
            source["patches"] = {p.name: integrity(p) for p in patches}
        (version_dir / "source.json").write_text(json.dumps(source, indent=4) + "\n")
        print(f"{module}@{version_dir.name}: {len(source['overlay'])} overlay files, {len(patches)} patches")


if __name__ == "__main__":
    main()
