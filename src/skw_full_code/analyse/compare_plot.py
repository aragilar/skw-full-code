# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import exp
import matplotlib.pyplot as plt

from disc_solver.utils import ODEIndex as DS_ODEIndex

from ..utils import ODEIndex

from .utils import (
    multiple_solution_plotter, analyse_main_wrapper_multisolution,
    analysis_func_wrapper_multisolution, common_plotting_options,
    get_common_plot_args, plot_output_wrapper, angles_to_heights,
    convert_ds_solution_to_skw,
)

plt.style.use("bmh")


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--v_z", choices=("log", "linear"), default="linear")
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "v_z_scale": args.get("v_z", "linear"),
    }


@analyse_main_wrapper_multisolution(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(solns, *, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return compare_plot(solns, **common_plot_args, **plot_args)


@analysis_func_wrapper_multisolution
def compare_plot(
    solns, *, plot_filename=None, show=False, linestyle='-', stop=90,
    figargs=None, v_z_scale="linear", title=None, close=True
):
    """
    Plot solutions to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        solns, linestyle=linestyle, stop=stop, figargs=figargs,
        v_z_scale=v_z_scale, title=title,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@multiple_solution_plotter
def generate_plot(
    solns, *, linestyle='-', stop=90, figargs=None, v_z_scale="linear"
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if figargs is None:
        figargs = {}

    fig, axes = plt.subplots(
        nrows=2, ncols=3, constrained_layout=True, sharex=True,
        gridspec_kw=dict(hspace=0), **figargs
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("height from plane ($z/h$)")
    axes = axes.flatten()

    for id_num, soln in enumerate(solns):
        solution = soln.solution
        if hasattr(soln, "heights"):
            heights = soln.heights
            cons = soln.initial_conditions
            B_r = solution[:, ODEIndex.b_r]
            B_φ = solution[:, ODEIndex.b_φ]
            v_r = solution[:, ODEIndex.w_r]
            v_φ_v_k = solution[:, ODEIndex.w_φ]
            v_z = cons.ρ_s / exp(solution[:, ODEIndex.ln_ρ])
            log_ρ = solution[:, ODEIndex.ln_ρ]
        else:
            inp = soln.solution_input
            heights = angles_to_heights(soln.angles, inp.c_s_on_v_k)
            solution = convert_ds_solution_to_skw(
                solution, c_s_on_v_k=inp.c_s_on_v_k, γ=inp.γ, heights=heights,
            )
            v_r = solution[:, DS_ODEIndex.v_r]
            v_φ_v_k = solution[:, DS_ODEIndex.v_φ]
            v_z = solution[:, DS_ODEIndex.v_θ]
            B_r = solution[:, DS_ODEIndex.B_r]
            B_φ = solution[:, DS_ODEIndex.B_φ]
            log_ρ = solution[:, DS_ODEIndex.ρ]

        indexes = heights <= stop

        param_names = [
            {
                "name": "$B_r/B_0$",
                "data": B_r,
            },
            {
                "name": "$B_φ/B_0$",
                "data": B_φ,
            },
            {
                "name": "$v_r/c_s$",
                "data": v_r,
            },
            {
                "name": "$(v_φ - v_k)/c_s$",
                "data": v_φ_v_k,
            },
            {
                "name": "$v_z/c_s$",
                "data": v_z,
                "scale": v_z_scale,
            },
            {
                "name": "$log(ρ/ρ_0)$",
                "data": log_ρ,
            },
        ]

        for i, settings in enumerate(param_names):
            ax = axes[i]
            ax.plot(
                heights[indexes], settings["data"][indexes], linestyle,
                label=str(id_num)
            )
            ax.set_ylabel(settings["name"])
            ax.set_yscale(settings.get("scale", "linear"))
    for ax in axes:
        ax.legend(loc=0)
    return fig
