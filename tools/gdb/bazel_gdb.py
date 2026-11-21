"""gdb helpers for debugging Bazel-built C++ binaries in this workspace.

Load it from ~/.gdbinit (works for gdb and CLion's bundled gdb):

    source /path/to/BaristaBot/tools/gdb/bazel_gdb.py

When gdb loads a binary that lives in a Bazel output tree, this:

1. Registers libc++'s own pretty printers from the LLVM toolchain Bazel
   downloaded (<output_base>/external/*llvm-project/libcxx/utils/gdb). They
   match the libc++ version the code was built with. CLion's bundled libc++
   renderers expect an older layout (e.g. unique_ptr::__value_,
   __hash_table::__p2_) and fail on LLVM 21. Ours register on the program
   space, which gdb consults before CLion's global printers.

2. Adds the binary's Bazel library directories (bin/_solib_*/*, and
   <binary>.runfiles/_main/lib) to solib-search-path. ROS tests run with
   LD_LIBRARY_PATH=lib, so libraries load under relative names such as
   lib/libhardware_interface.so that gdb cannot open on its own.
"""

import glob
import os
import sys

import gdb
import gdb.printing

_handled = set()


def _bazel_layout(exe):
    """(real path, output base, bin dir) for a binary in a Bazel output tree."""
    real = os.path.realpath(exe)
    if "/execroot/" not in real or "/bazel-out/" not in real:
        return None
    output_base = real.split("/execroot/", 1)[0]
    head, _, rest = real.partition("/bazel-out/")
    config = rest.split("/", 1)[0]
    return real, output_base, os.path.join(head, "bazel-out", config, "bin")


def _register_libcxx_printers(output_base):
    dirs = sorted(glob.glob(os.path.join(output_base, "external", "*llvm-project", "libcxx", "utils", "gdb")))
    if not dirs:
        return
    if dirs[0] not in sys.path:
        sys.path.insert(0, dirs[0])
    try:
        from libcxx import printers
    except Exception as e:  # pragma: no cover - only reached on odd setups
        print(f"bazel_gdb: could not import libc++ printers from {dirs[0]}: {e}")
        return
    if not getattr(printers, "_bazel_gdb_loader", False):
        printers.register_libcxx_printer_loader()
        printers._bazel_gdb_loader = True
    # The loader only fires for objfiles loaded later; register for this one now.
    progspace = gdb.current_progspace()
    if not getattr(progspace, printers._libcxx_printer_name, False):
        gdb.printing.register_pretty_printer(progspace, printers.LibcxxPrettyPrinter(printers._libcxx_printer_name))
        setattr(progspace, printers._libcxx_printer_name, True)


def _add_solib_dirs(real, bin_dir):
    dirs = sorted(d for d in glob.glob(os.path.join(bin_dir, "_solib_*", "*")) if os.path.isdir(d))
    runfiles_lib = real + ".runfiles/_main/lib"
    if os.path.isdir(runfiles_lib):
        dirs.insert(0, runfiles_lib)
    if not dirs:
        return
    current = gdb.parameter("solib-search-path") or ""
    have = set(p for p in current.split(":") if p)
    new = [d for d in dirs if d not in have]
    if new:
        gdb.execute("set solib-search-path " + ":".join(([current] if current else []) + new), to_string=True)


def _setup(exe):
    layout = _bazel_layout(exe) if exe else None
    if not layout or layout[0] in _handled:
        return
    real, output_base, bin_dir = layout
    _handled.add(real)
    _register_libcxx_printers(output_base)
    _add_solib_dirs(real, bin_dir)
    print(f"bazel_gdb: libc++ printers and Bazel library paths set up for {os.path.basename(real)}")


def _on_new_objfile(event):
    main = gdb.current_progspace().filename
    if main and event.new_objfile.filename == main:
        _setup(main)


gdb.events.new_objfile.connect(_on_new_objfile)
_setup(gdb.current_progspace().filename)  # binary already loaded when sourced late
