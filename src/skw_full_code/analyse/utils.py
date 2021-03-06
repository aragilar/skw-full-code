# -*- coding: utf-8 -*-
"""
Utils for analysis code
"""
import argparse
from functools import wraps
from os import fspath
import sys

from logbook.compat import redirected_warnings, redirected_logging

from numpy import zeros, log, sin, arcsin, sqrt
from matplotlib.colors import TABLEAU_COLORS, XKCD_COLORS
import matplotlib.pyplot as plt

from h5preserve import open as h5open

from disc_solver.file_format import registries as ds_registries
from disc_solver.utils import ODEIndex as DS_ODEIndex

from .. import __version__ as skw_version
from ..file_format import registries
from ..logging import log_handler, logging_options
from ..utils import str_to_float, get_solutions, SKWError

ds_and_skw_registries = registries + ds_registries


def single_solution_plotter(func):
    """
    Pulls out common elements of plots which take a single solution
    """
    @wraps(func)
    def plot_wrapper(h5file, solution, *args, title=None, **kwargs):
        """
        Wraps plot functions
        """
        if solution is None:
            solution = "0"
        fig = func(
            get_solutions(h5file, solution), *args, **kwargs
        )
        if title is None:
            fig.suptitle("{}:{}:{}".format(
                h5file.config_filename, h5file.config_input.label, solution
            ))
        else:
            fig.suptitle(title)
        return fig
    return plot_wrapper


def multiple_solution_plotter(func):
    """
    Pulls out common elements of plots which take multiple solutions
    """
    @wraps(func)
    def plot_wrapper(solution_pairs, *args, title=None, **kwargs):
        """
        Wraps plot functions
        """
        title_list = []

        def solution_loader(pairs):
            for run, name in pairs:
                title_list.append("{}:{}:{}".format(
                    run.config_filename, run.config_input.label, name
                ))
                yield get_solutions(run, name)

        fig = func(solution_loader(solution_pairs), *args, **kwargs)
        if title is None:
            fig.suptitle('\n'.join(title_list))
        else:
            fig.suptitle(title)
        return fig
    return plot_wrapper


def analysis_func_wrapper_multisolution(func):
    """
    Wrapper for main analysis functions which take multiple solutions
    """
    @wraps(func)
    def wrapper(solutions_pairs, *args, **kwargs):
        """
        wrapper for analysis_func_wrapper_multisolution
        """
        def file_loader(pairs):
            for filename, index_str in pairs:
                with h5open(filename, ds_and_skw_registries) as soln_file:
                    yield soln_file["run"], index_str
        return func(file_loader(solutions_pairs), *args, **kwargs)

    return wrapper


def common_plotting_options(parser):
    """
    Common plotting options to control appearance and output
    """
    parser.add_argument("--show", action="store_true", default=False)
    parser.add_argument("--filename")
    parser.add_argument("--figsize", nargs=2)
    parser.add_argument("--stop")
    parser.add_argument("--linestyle")
    parser.add_argument("--title")
    return parser


def get_common_plot_args(args):
    """
    Extract the common plot options into the correct variables
    """
    figargs = {}
    if args.get("figsize") is not None:
        figargs["figsize"] = (
            int(args["figsize"][0]),
            int(args["figsize"][1])
        )

    return {
        "show": args.get("show", False),
        "plot_filename": args.get("filename"),
        "figargs": figargs,
        "stop": str_to_float(args.get("stop", "90")),
        "linestyle": args.get("linestyle", "-"),
        "title": args.get("title"),
    }


def analyse_main_wrapper(
    cmd_description, cmd_parser_func, cmd_parser_splitters=None
):
    """
    Wrapper for main cmd for analysis cmds
    """
    if cmd_parser_splitters is None:
        cmd_parser_splitters = {}

    def decorator(cmd):
        """
        decorator for analyse_main_wrapper
        """
        @wraps(cmd)
        def wrap_analysis_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys.argv[1:]
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument(
                '--version', action='version',
                version='%(prog)s ' + skw_version
            )
            parser.add_argument("soln_file")
            parser.add_argument("soln_range")
            logging_options(parser)
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    return cmd(
                        args["soln_file"],
                        soln_range=args["soln_range"],
                        **cmd_args
                    )

        return wrap_analysis_main

    return decorator


def analyse_main_wrapper_multisolution(
    cmd_description, cmd_parser_func, cmd_parser_splitters=None
):
    """
    Wrapper for main cmd for analysis cmds
    """
    if cmd_parser_splitters is None:
        cmd_parser_splitters = {}

    def decorator(cmd):
        """
        decorator for analyse_main_wrapper_multisolution
        """
        @wraps(cmd)
        def wrap_analysis_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys.argv[1:]
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument(
                '--version', action='version',
                version='%(prog)s ' + skw_version
            )
            parser.add_argument(
                "-r", "--runs", action="append", nargs=2,
                metavar=("soln_file", "soln_range"), dest="soln_pairs",
            )
            logging_options(parser)
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    return cmd(args["soln_pairs"], **cmd_args)

        return wrap_analysis_main

    return decorator


def analyse_multisolution_wrapper(
    cmd_description, cmd_parser_func, cmd_parser_splitters=None
):
    """
    Wrapper for main entry point for analysis functions which use multiple
    solutions
    """
    def decorator(cmd):
        """
        decorator for analyse_main_wrapper
        """
        @wraps(cmd)
        def wrap_analysis_main(argv=None):
            """
            Actual main function for analysis code, deals with parsers and
            logging
            """
            if argv is None:
                argv = sys.argv[1:]
            cmd_args = {}
            parser = argparse.ArgumentParser(
                description=cmd_description,
                argument_default=argparse.SUPPRESS,
            )
            parser.add_argument(
                '--version', action='version',
                version='%(prog)s ' + skw_version
            )
            parser.add_argument("soln_file")
            logging_options(parser)
            cmd_parser = cmd_parser_func(parser)
            args = vars(cmd_parser.parse_args(argv))
            for name, func in cmd_parser_splitters.items():
                cmd_args[name] = func(args)
            with log_handler(args):
                with redirected_warnings(), redirected_logging():
                    with h5open(args["soln_file"], registries) as soln_file:
                        solutions = soln_file["run"].solutions.values()
                        return cmd(
                            *solutions, **cmd_args
                        )

        return wrap_analysis_main

    return decorator


def analysis_func_wrapper(func):
    """
    Wrapper for main analysis functions which have common elements
    """
    @wraps(func)
    def wrapper(filename, *args, **kwargs):
        """
        wrapper for analysis_func_wrapper
        """
        with h5open(filename, registries) as soln_file:
            return func(soln_file["run"], *args, **kwargs)
    return wrapper


class AnalysisError(SKWError):
    """
    Error class for problems with analysis routines
    """
    pass


def distinct_color_map(size):
    """
    Generate a list of unique colours for matplotlib line/scatter plots
    """
    if size > len(TABLEAU_COLORS):
        return XKCD_COLORS.keys()
    return TABLEAU_COLORS.keys()


def plot_output_wrapper(
    fig, *, file=None, show=False, close=True, facecolor="none", **kwargs
):
    """
    Wrapper for handling whether a figure is shown, saved to file, and/or
    closed.
    """
    if file is not None:
        fig.savefig(fspath(file), facecolor=facecolor, **kwargs)
    if show:
        plt.show()
    if close:
        plt.close(fig)
        return None
    return fig


def convert_ds_solution_to_skw(solution, *, c_s_on_v_k, γ, heights):
    """
    Convert a solution from disc-solver to skw variables.
    """
    scale_factor = sqrt(1 + heights ** 2 * c_s_on_v_k ** 2)
    v_scale = 1 / sqrt(scale_factor)
    B_scale = scale_factor ** (γ - 5 / 4)
    ρ_scale = scale_factor ** (2 * γ - 3 / 2)
    skw_solution = zeros(solution.shape)
    skw_solution[:, DS_ODEIndex.v_r] = v_scale * solution[:, DS_ODEIndex.v_r]
    skw_solution[:, DS_ODEIndex.v_φ] = solution[:, DS_ODEIndex.v_φ] - (
        1 / c_s_on_v_k
    ) * v_scale
    skw_solution[:, DS_ODEIndex.v_θ] = v_scale * solution[:, DS_ODEIndex.v_θ]
    skw_solution[:, DS_ODEIndex.B_r] = B_scale * solution[:, DS_ODEIndex.B_r]
    skw_solution[:, DS_ODEIndex.B_φ] = B_scale * solution[:, DS_ODEIndex.B_φ]
    skw_solution[:, DS_ODEIndex.ρ] = log(ρ_scale * solution[:, DS_ODEIndex.ρ])
    return skw_solution


def heights_to_angles(heights, c_s_on_v_k):
    """
    Convert heights in skw space to angles in disc-solver space.
    """
    return arcsin(heights * c_s_on_v_k)


def angles_to_heights(angles, c_s_on_v_k):
    """
    Convert angles in disc-solver space to heights in skw space.
    """
    return sin(angles) / c_s_on_v_k
