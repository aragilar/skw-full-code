# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import sqrt, ones as np_ones, exp
import matplotlib.pyplot as plt

from ..utils import (
    mhd_wave_speeds, MHD_Wave_Index, ODEIndex, MAGNETIC_INDEXES,
)

from .utils import (
    single_solution_plotter, analyse_main_wrapper, analysis_func_wrapper,
    common_plotting_options, get_common_plot_args, plot_output_wrapper,
)

plt.style.use("bmh")


def plot_parser(parser):
    """
    Add arguments for plot command to parser
    """
    common_plotting_options(parser)
    parser.add_argument("--v_z", choices=("log", "linear"), default="linear")
    parser.add_argument(
        "--with-slow", action='store_true', default=False)
    parser.add_argument(
        "--with-alfven", action='store_true', default=False)
    parser.add_argument(
        "--with-fast", action='store_true', default=False)
    parser.add_argument(
        "--with-sonic", action='store_true', default=False)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "v_z_scale": args.get("v_z", "linear"),
        "with_slow": args.get("with_slow", False),
        "with_alfven": args.get("with_alfven", False),
        "with_fast": args.get("with_fast", False),
        "with_sonic": args.get("with_sonic", False),
    }


@analyse_main_wrapper(
    "Main plotter for DiscSolver",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for ds-plot
    """
    return plot(soln, soln_range=soln_range, **common_plot_args, **plot_args)


@analysis_func_wrapper
def plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    with_slow=False, with_alfven=False, with_fast=False, with_sonic=False,
    stop=90, figargs=None, v_z_scale="linear", title=None, close=True
):
    """
    Plot solution to file
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, with_slow=with_slow,
        with_alfven=with_alfven, with_fast=with_fast, with_sonic=with_sonic,
        stop=stop, figargs=figargs, v_z_scale=v_z_scale, title=title,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    soln, *, linestyle='-', with_slow=False, with_alfven=False,
    with_fast=False, with_sonic=False, stop=90, figargs=None,
    v_z_scale="linear"
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if figargs is None:
        figargs = {}

    solution = soln.solution
    heights = soln.heights
    cons = soln.initial_conditions

    wave_speeds = sqrt(mhd_wave_speeds(
        solution[:, MAGNETIC_INDEXES], solution[:, ODEIndex.ln_ρ], 1
    ))

    indexes = heights <= stop

    param_names = [
        {
            "name": "$B_r/B_0$",
            "data": solution[:, ODEIndex.b_r],
        },
        {
            "name": "$B_φ/B_0$",
            "data": solution[:, ODEIndex.b_φ],
        },
        {
            "name": "$v_r/c_s$",
            "data": solution[:, ODEIndex.w_r],
        },
        {
            "name": "$(v_φ - v_k)/c_s$",
            "data": solution[:, ODEIndex.w_φ],
        },
        {
            "name": "$v_z/c_s$",
            "data": cons.ρ_s / exp(solution[:, ODEIndex.ln_ρ]),
            "legend": True,
            "scale": v_z_scale,
            "extras": []
        },
        {
            "name": "$log(ρ/ρ_0)$",
            "data": solution[:, ODEIndex.ln_ρ],
        },
    ]

    if with_slow:
        param_names[5]["extras"].append({
            "label": "slow",
            "data": wave_speeds[MHD_Wave_Index.slow],
        })
    if with_alfven:
        param_names[5]["extras"].append({
            "label": "alfven",
            "data": wave_speeds[MHD_Wave_Index.alfven],
        })
    if with_fast:
        param_names[5]["extras"].append({
            "label": "fast",
            "data": wave_speeds[MHD_Wave_Index.fast],
        })
    if with_sonic:
        param_names[5]["extras"].append({
            "label": "sound",
            "data": np_ones(len(solution)),
        })

    fig, axes = plt.subplots(
        nrows=2, ncols=3, constrained_layout=True, sharex=True,
        gridspec_kw=dict(hspace=0), **figargs
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("height from plane ($z/h$)")

    axes.shape = len(param_names)
    for i, settings in enumerate(param_names):
        ax = axes[i]
        ax.plot(
            heights[indexes],
            settings["data"][indexes], linestyle, label=settings["name"]
        )
        for extra in settings.get("extras", []):
            ax.plot(
                heights[indexes],
                extra["data"][indexes],
                label=extra.get("label")
            )
        ax.set_ylabel(settings["name"])
        ax.set_yscale(settings.get("scale", "linear"))
        if settings.get("legend"):
            ax.legend(loc=0)
    return fig
