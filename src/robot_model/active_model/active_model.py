"""The active robot model: the one every app uses unless given --model.

It is the model named in active_model.txt at the workspace root (checked in: edit it to switch every app
to another model).

    ap = argparse.ArgumentParser()
    active_model.add_model_argument(ap, PLANNERS)   # --model, default NAME, one of PLANNERS
"""

import argparse
from collections.abc import Iterable

from src.robot_model.active_model.active_model_name import NAME

__all__ = ["NAME", "add_model_argument"]


def add_model_argument(parser: argparse.ArgumentParser, choices: Iterable[str], help: str = "robot model") -> argparse.Action:
    """Adds --model, one of `choices`, defaulting to the active model.

    An active model that isn't one of `choices` is an error only when --model isn't given."""
    choices = sorted(choices)

    def model(name: str) -> str:  # argparse also applies it to the (string) default
        if name in choices:
            return name
        source = " (the active model, from active_model.txt: pass --model)" if name == NAME else ""
        raise argparse.ArgumentTypeError(f"{name!r}{source} is not one of {', '.join(choices)}")

    return parser.add_argument(
        "--model",
        type=model,
        default=NAME,
        metavar="{" + ",".join(choices) + "}",
        help=f"{help} (default: the active model in active_model.txt, {NAME})",
    )
