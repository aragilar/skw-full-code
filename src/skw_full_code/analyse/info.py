# -*- coding: utf-8 -*-
"""
Info command and associated code
"""
from sys import stdout

from ..utils import get_solutions
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
        "run", "status", "input", "initial-conditions", "solutions",
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
        if group == "input":
            print("input settings:", file=output_file)
            for name, value in vars(inp).items():
                print(INPUT_FORMAT.format(name, value), file=output_file)
        elif group == "initial-conditions":
            print("initial conditions:", file=output_file)
            for name, value in vars(init_con).items():
                print(INIT_FORMAT.format(name, value), file=output_file)
        else:
            raise AnalysisError("Cannot find {}.".format(group))
