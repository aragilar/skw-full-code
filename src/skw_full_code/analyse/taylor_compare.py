# -*- coding: utf-8 -*-
"""
Plot command for DiscSolver
"""
from numpy import zeros, log, outer, ones, newaxis, arcsin
import matplotlib.pyplot as plt

from disc_solver.file_format import SolutionInput as DSSolutionInput
from disc_solver.solve.taylor_space import (
    compute_taylor_values as ds_compute_taylor_values,
)
from disc_solver.utils import ODEIndex as DS_ODEIndex

from ..utils import ODEIndex

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
    parser.add_argument("--c_s_on_v_k", type=float, default=0.05)
    parser.add_argument("--γ", type=float, default=1e-7)
    parser.add_argument("--show-both", action='store_true', default=False)
    return parser


def get_plot_args(args):
    """
    Parse plot args
    """
    return {
        "c_s_on_v_k": args.get("c_s_on_v_k", 0.05),
        "γ": args.get("γ", 1e-7),
        "show_both": args.get("show_both", False),
    }


@analyse_main_wrapper(
    "Compare taylor solution with skw solution",
    plot_parser,
    cmd_parser_splitters={
        "common_plot_args": get_common_plot_args,
        "plot_args": get_plot_args,
    }
)
def plot_main(soln, *, soln_range, common_plot_args, plot_args):
    """
    Entry point for skw-main-taylor-compare
    """
    return plot(soln, soln_range=soln_range, **common_plot_args, **plot_args)


@analysis_func_wrapper
def plot(
    soln, *, soln_range=None, plot_filename=None, show=False, linestyle='-',
    stop=90, figargs=None, title=None, close=True, c_s_on_v_k=0.05, γ=1e-7,
    show_both=False
):
    """
    Plot difference between taylor solution and skw solution
    """
    # pylint: disable=too-many-function-args,unexpected-keyword-arg
    fig = generate_plot(
        soln, soln_range, linestyle=linestyle, stop=stop, figargs=figargs,
        title=title, γ=γ, c_s_on_v_k=c_s_on_v_k, show_both=show_both,
    )

    return plot_output_wrapper(
        fig, file=plot_filename, show=show, close=close
    )


@single_solution_plotter
def generate_plot(
    soln, *, linestyle='-', stop=90, figargs=None, c_s_on_v_k=0.05, γ=1e-7,
    show_both=False
):
    """
    Generate plot, with enough freedom to be able to format fig
    """
    if figargs is None:
        figargs = {}

    solution = soln.solution
    heights = soln.heights

    indexes = heights <= stop

    taylor_solutions = compute_taylor(
        soln.solution_input, heights, γ=γ, c_s_on_v_k=c_s_on_v_k,
    )
    diff_solution = solution - taylor_solutions

    param_names = [
        {
            "name": "$B_r/B_0$",
            "data": diff_solution[:, ODEIndex.b_r],
            "index": ODEIndex.b_r,
        },
        {
            "name": "$B_φ/B_0$",
            "data": diff_solution[:, ODEIndex.b_φ],
            "index": ODEIndex.b_φ,
        },
        {
            "name": "$v_r/c_s$",
            "data": diff_solution[:, ODEIndex.w_r],
            "index": ODEIndex.w_r,
        },
        {
            "name": "$(v_φ - v_k)/c_s$",
            "data": diff_solution[:, ODEIndex.w_φ],
            "index": ODEIndex.w_φ,
        },
        {
            "name": "$log(ρ/ρ_0)$",
            "data": diff_solution[:, ODEIndex.ln_ρ],
            "index": ODEIndex.ln_ρ,
        },
    ]

    fig, axes = plt.subplots(
        nrows=2, ncols=3, constrained_layout=True, sharex=True,
        gridspec_kw=dict(hspace=0), **figargs
    )

    # only add label to bottom plots
    for ax in axes[1]:
        ax.set_xlabel("$z/z_s$")

    axes.shape = axes.size
    for i, settings in enumerate(param_names):
        ax = axes[i]
        if show_both:
            ax.plot(
                heights[indexes],
                solution[indexes, settings["index"]], linestyle, label="SKW"
            )
            ax.plot(
                heights[indexes],
                taylor_solutions[indexes, settings["index"]], linestyle,
                label="Taylor"
            )
            ax.legend(loc=0)
        else:
            ax.plot(
                heights[indexes],
                settings["data"][indexes], linestyle, label=settings["name"]
            )
        ax.set_ylabel(settings["name"])
        ax.set_yscale(settings.get("scale", "linear"))
    return fig


def compute_taylor(skw_config, heights, c_s_on_v_k=0.05, γ=1e-7):
    """
    Compute solution based on taylor series from disc-solver
    """
    angles = arcsin(heights * c_s_on_v_k)

    def sum_taylor(coef, count=0):
        if not coef:
            return zeros(angles.shape)
        if len(coef) == 1:
            return outer(ones(angles.shape), coef[0])
        divisor = count if count != 0 else 1
        return (
            sum_taylor(coef[1:], count=count+1) * angles[:, newaxis] + coef[0]
        ) / divisor

    def convert_ds_solution(solution):
        skw_solution = zeros([solution.shape[0], len(ODEIndex)])
        skw_solution[:, ODEIndex.w_r] = solution[:, DS_ODEIndex.v_r]
        skw_solution[:, ODEIndex.w_φ] = solution[:, DS_ODEIndex.v_φ] - (
            1 / c_s_on_v_k
        )
        skw_solution[:, ODEIndex.b_r] = solution[:, DS_ODEIndex.B_r]
        skw_solution[:, ODEIndex.b_φ] = solution[:, DS_ODEIndex.B_φ]
        skw_solution[:, ODEIndex.ln_ρ] = log(solution[:, DS_ODEIndex.ρ])
        return skw_solution

    σ_O_0 = skw_config.σ_O_0
    σ_P_0 = skw_config.σ_P_0
    σ_H_0 = skw_config.σ_H_0
    η_O = c_s_on_v_k / σ_O_0
    η_A = c_s_on_v_k * (σ_P_0 / (σ_P_0**2 + σ_H_0**2) - 1 / σ_O_0)
    η_H = c_s_on_v_k * (σ_H_0 / (σ_P_0**2 + σ_H_0**2))

    ds_soln_input = DSSolutionInput(
        start=skw_config.start,
        stop=skw_config.stop,
        taylor_stop_angle=None,
        max_steps=None,
        num_angles=len(angles),
        relative_tolerance=skw_config.relative_tolerance,
        absolute_tolerance=skw_config.absolute_tolerance,
        jump_before_sonic=None,
        η_derivs=False,
        nwalkers=None,
        iterations=None,
        threads=None,
        target_velocity=None,
        split_method=None,
        use_taylor_jump=None,
        v_rin_on_c_s=skw_config.v_rin_on_c_s,
        v_a_on_c_s=skw_config.v_a_on_c_s,
        η_O=η_O,
        η_H=η_H,
        η_A=η_A,
        γ=γ,
        c_s_on_v_k=c_s_on_v_k,
    )
    return convert_ds_solution(
        sum_taylor(ds_compute_taylor_values(ds_soln_input))
    )
