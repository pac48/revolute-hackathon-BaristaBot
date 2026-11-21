"""Command-line argument parsing shared by move_to, plan_to and move_cartesian."""

import argparse


def parse_goal(items: list[str]) -> dict[str, float]:
    goal = {}
    for item in items:
        joint, sep, value = item.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected JOINT=POSITION, got {item!r}")
        goal[joint] = float(value)
    return goal


def numbers(text: str, counts: tuple[int, ...]) -> list[float]:
    values = [float(v) for v in text.split(",")]
    if len(values) not in counts:
        raise argparse.ArgumentTypeError(f"expected {' or '.join(map(str, counts))} comma-separated numbers")
    return values


def attach_values(argv: list[str], options: tuple[str, ...]) -> list[str]:
    """Rewrites `--box -0.1,...` to `--box=-0.1,...`: argparse takes a value starting with '-' for an option."""
    out = []
    for arg in argv:
        if out and out[-1] in options and arg.startswith("-") and not arg.startswith("--"):
            out[-1] = f"{out[-1]}={arg}"
        else:
            out.append(arg)
    return out
