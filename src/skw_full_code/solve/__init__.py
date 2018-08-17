# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging

from h5preserve import open as h5open

from .config import (
    get_input_from_conffile, config_input_to_soln_input, define_conditions,
)
from .solution import solution
from .utils import add_solver_arguments, validate_overrides

from .. import __version__ as skw_full_code_version
from ..file_format import registries, Run
from ..float_handling import float_type
from ..logging import log_handler
from ..utils import expanded_path

log = logbook.Logger(__name__)


def solver(inp, run):
    """
    Single run solver
    """
    single_solution = solution(inp, define_conditions(inp))
    run.solutions["0"] = single_solution
    run.final_solution = run.solutions["0"]


def solve(
    *, output_file, config_file, output_dir, overrides=None
):
    """
    Main function to generate solution
    """
    config_input = get_input_from_conffile(
        config_file=config_file, overrides=overrides
    )
    run = Run(
        config_input=config_input,
        config_filename=str(config_file),
        skw_full_code_version=skw_full_code_version,
        float_type=str(float_type),
    )

    if output_file is None:
        output_file = Path(config_input.label + str(arrow.now()) + ".hdf5")
    output_file = expanded_path(output_dir / output_file)

    with h5open(output_file, registries) as f:
        f["run"] = run
        solver(config_input_to_soln_input(config_input), run)

    return output_file


def main():
    """
    Entry point for skw-full-soln
    """
    parser = argparse.ArgumentParser(description='Solver for skw_full_code')
    add_solver_arguments(parser)
    parser.add_argument("config_file")

    args = vars(parser.parse_args())

    config_file = expanded_path(args["config_file"])
    output_dir = expanded_path(args["output_dir"])
    output_file = args.get("output_file", None)
    overrides = validate_overrides(args.get("override", []))

    with log_handler(args), redirected_warnings(), redirected_logging():
        print(solve(
            output_file=output_file, config_file=config_file,
            output_dir=output_dir, overrides=overrides,
        ))
