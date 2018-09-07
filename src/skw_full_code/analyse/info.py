# -*- coding: utf-8 -*-
"""
Info command and associated code
"""
from sys import stdout

import numpy as np

from ..utils import (
    is_supersonic, find_in_array, ODEIndex, MAGNETIC_INDEXES, get_solutions,
)
from .utils import (
    analyse_main_wrapper, analysis_func_wrapper, AnalysisError,
)

INPUT_FORMAT = " {: <20}: {}"
INIT_FORMAT = " {: <20}: {}"
OTHER_FORMAT = " {: <20}: {}"


def info_parser(parser):
    """
    Add arguments for info command to parser
    """
    parser.add_argument("group", choices=[
        "run", "status", "input", "initial-conditions", "sonic-points",
        "crosses-points", "sonic-on-scale", "solutions",
    ])
    return parser


@analyse_main_wrapper(
    "Info dumper for output from DiscSolver",
    info_parser,
    cmd_parser_splitters={
        "group": lambda args: args["group"]
    }
)
def info_main(soln_file, *, group, soln_range):
    """
    Entry point for ds-info
    """
    return info(
        soln_file, group=group, soln_range=soln_range, output_file=stdout
    )


@analysis_func_wrapper
def info(soln_file, *, group, soln_range, output_file):
    """
    Output info about the solution
    """
    soln_instance = get_solutions(soln_file, soln_range)
    if group == "run":
        print("run properties:", file=output_file)
        print(
            "label: {}".format(soln_file.config_input.label),
            file=output_file,
        )
        print(
            "config filename: {}".format(soln_file.config_filename),
            file=output_file
        )
        print(
            "number of solutions: {}".format(len(soln_file.solutions)),
            file=output_file
        )
        print("config:", file=output_file)
        for name, value in vars(soln_file.config_input).items():
            print(INPUT_FORMAT.format(name, value), file=output_file)
    elif group == "status":
        print(
            "ODE return flag: {!s}".format(soln_instance.flag),
            file=output_file
        )
    elif group == "solutions":
        for name in soln_file.solutions:
            print(name, file=output_file)
    else:
        inp = soln_instance.solution_input
        init_con = soln_instance.initial_conditions
        c_s = 1
        if group == "input":
            print("input settings:", file=output_file)
            for name, value in vars(inp).items():
                print(INPUT_FORMAT.format(name, value), file=output_file)
        elif group == "initial-conditions":
            print("initial conditions:", file=output_file)
            for name, value in vars(init_con).items():
                print(INIT_FORMAT.format(name, value), file=output_file)
        else:
            pass
            #soln = soln_instance.solution
            #heights = soln_instance.heights
            #zero_soln = np.zeros(len(soln))
            #v = np.array([zero_soln, zero_soln, soln[:, ODEIndex.v_θ]])
            #slow_index = find_in_array(is_supersonic(
            #    v.T, soln[:, MAGNETIC_INDEXES], soln[:, ODEIndex.ρ],
            #    c_s, "slow"
            #), True)
            #alfven_index = find_in_array(is_supersonic(
            #    v.T, soln[:, MAGNETIC_INDEXES], soln[:, ODEIndex.ρ],
            #    c_s, "alfven"
            #), True)
            #fast_index = find_in_array(is_supersonic(
            #    v.T, soln[:, MAGNETIC_INDEXES], soln[:, ODEIndex.ρ],
            #    c_s, "fast"
            #), True)

            #if group == "crosses-points":
            #    if fast_index:
            #        print(
            #            "{}: fast".format(soln_file.config_input.label),
            #            file=output_file
            #        )
            #    elif alfven_index:
            #        print(
            #            "{}: alfven".format(soln_file.config_input.label),
            #            file=output_file
            #        )
            #    elif slow_index:
            #        print(
            #            "{}: slow".format(soln_file.config_input.label),
            #            file=output_file
            #        )
            #    else:
            #        print(
            #            "{}: none".format(soln_file.config_input.label),
            #            file=output_file
            #        )
            #elif group == "sonic-points":
            #    print(OTHER_FORMAT.format(
            #        "slow sonic point",
            #        heights[slow_index] if slow_index else None
            #    ), file=output_file)
            #    print(OTHER_FORMAT.format(
            #        "alfven sonic point",
            #        heights[alfven_index] if alfven_index else None
            #    ), file=output_file)
            #    print(OTHER_FORMAT.format(
            #        "fast sonic point",
            #        heights[fast_index] if fast_index else None
            #    ), file=output_file)
            #else:
            #    raise AnalysisError("Cannot find {}.".format(group))