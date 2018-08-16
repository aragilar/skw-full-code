# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import arrow
import logbook
from logbook.compat import redirected_warnings, redirected_logging
from numpy import sqrt, exp
from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from h5preserve import open as h5open

from .config import (
    get_input_from_conffile, config_input_to_soln_input
)
from .utils import add_solver_arguments, SolverError, validate_overrides

from .. import __version__ as ds_version
from ..file_format import registries, Run
from ..float_handling import float_type
from ..logging import log_handler
from ..utils import expanded_path


from .utils import (
    error_handler, SolverError,
)

from ..float_handling import float_type
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "dense"
SONIC_POINT_TOLERANCE = float_type(0.01)

log = logbook.Logger(__name__)


def ode_system(*, a_0, σ_O_0, σ_P_0, σ_H_0, ρ_s, z_s):
    def rhs_func(z, variables, derivs):
        w_r = variables[ODEIndex.w_r]
        w_φ = variables[ODEIndex.w_φ]
        w_Er = variables[ODEIndex.w_Er]
        b_φ = variables[ODEIndex.b_φ]
        b_r = variables[ODEIndex.b_r]
        ln_ρ = variables[ODEIndex.ln_ρ]

        ρ = exp(ln_ρ)
        w_z = ρ_s / ρ
        b_sq = 1 + b_φ**2 + b_r**2
        b = sqrt(b_sq)

        σ_scale = ρ / b_sq
        σ_O = σ_O_0 * σ_scale
        σ_P = σ_P_0 * σ_scale
        σ_H = σ_H_0 * σ_scale

        e_r_prime = w_Er + w_φ - w_z * b_φ
        e_φ_prime = w_z * b_r - w_r

        e_z_prime = (
            - (e_r_prime*b_r + e_φ_prime * b_φ) * (σ_O - σ_P) + σ_H * b * (
                e_r_prime * b_φ - e_φ_prime * b_r
            )
        ) / (σ_O - σ_P + b_sq * σ_P)

        y = e_z_prime + b_r * e_r_prime + b_φ * e_φ_prime

        j_r = y * (σ_O - σ_P) * b_r + σ_H / b * (
            e_z_prime * b_φ - e_φ_prime
        ) + σ_P * e_r_prime

        j_φ = y * (σ_O - σ_P) * b_φ + σ_H / b * (
            e_r_prime - e_z_prime * b_r
        ) + σ_P * e_φ_prime

        deriv_w_Er = - float_type(3) / float_type(2) * b_r
        deriv_b_r = j_φ
        deriv_b_φ = - j_r
        deriv_w_r = (a_0 ** 2 / ρ * j_φ + 2 * w_φ) / w_z
        deriv_w_φ = - (a_0 ** 2 / ρ * j_r + w_r / 2) / w_z
        deriv_ln_ρ = (a_0 ** 2 / ρ * (j_r * b_φ - j_φ * b_r) - z * z_s) / (
            (1 + w_z) * (1 - w_z)
        )

        derivs[ODEIndex.w_r] = deriv_w_r
        derivs[ODEIndex.w_φ] = deriv_w_φ
        derivs[ODEIndex.w_Er] = deriv_w_Er
        derivs[ODEIndex.b_φ] = deriv_b_φ
        derivs[ODEIndex.b_r] = deriv_b_r
        derivs[ODEIndex.ln_ρ] = deriv_ln_ρ

        return 0
    return rhs_func


def main_solution(
    *, heights, initial_conditions, a_0, σ_O_0, σ_P_0, σ_H_0, ρ_s, z_s,
    relative_tolerance=float_type(1e-6), absolute_tolerance=float_type(1e-10),
    max_steps=500, onroot_func=None, tstop=None, ontstop_func=None,
    root_func=None, root_func_args=None
):
    """
    Find solution
    """
    extra_args = {}
    if root_func is not None:
        extra_args["rootfn"] = root_func
        if root_func_args is not None:
            extra_args["nr_rootfns"] = root_func_args
        else:
            raise SolverError("Need to specify size of root array")

    system = ode_system(
        a_0=a_0, σ_O_0=σ_O_0, σ_P_0=σ_P_0, σ_H_0=σ_H_0, ρ_s=ρ_s, z_s=z_s,
    )

    solver = ode(
        INTEGRATOR, system,
        linsolver=LINSOLVER,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        max_steps=max_steps,
        validate_flags=True,
        old_api=False,
        err_handler=error_handler,
        onroot=onroot_func,
        tstop=tstop,
        ontstop=ontstop_func,
        bdf_stability_detection=True,
        **extra_args
    )

    try:
        soln = solver.solve(heights, initial_conditions)
    except CVODESolveFailed as e:
        soln = e.soln
        log.warn(
            "Solver stopped at {} with flag {!s}.\n{}".format(
                soln.errors.t,
                soln.flag, soln.message
            )
        )
        if soln.flag == StatusEnum.TOO_CLOSE:
            raise e
    except CVODESolveFoundRoot as e:
        soln = e.soln
        log.notice("Found root at {}".format(soln.roots.t))
    except CVODESolveReachedTSTOP as e:
        soln = e.soln
        for tstop_scaled in soln.tstop.t:
            log.notice("Stopped at {}".format(tstop_scaled))

    return soln




log = logbook.Logger(__name__)


def solve(
    *, output_file, sonic_method, config_file, output_dir, store_internal,
    overrides=None
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
        disc_solver_version=ds_version,
        float_type=str(float_type),
        sonic_method=sonic_method,
    )

    if output_file is None:
        output_file = Path(config_input.label + str(arrow.now()) + ".hdf5")
    output_file = expanded_path(output_dir / output_file)

    with h5open(output_file, registries) as f:
        f["run"] = run
        if sonic_method == "step":
            stepper_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        elif sonic_method == "single":
            single_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        elif sonic_method == "dae_single":
            dae_single_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        elif sonic_method == "mcmc":
            mcmc_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        elif sonic_method == "sonic_root":
            sonic_root_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        elif sonic_method == "hydrostatic":
            hydrostatic_solver(
                config_input_to_soln_input(config_input), run,
                store_internal=store_internal,
            )
        else:
            raise SolverError("No method chosen to cross sonic point")

    return output_file


def main():
    """
    Entry point for ds-soln
    """
    parser = argparse.ArgumentParser(description='Solver for DiscSolver')
    add_solver_arguments(parser)
    parser.add_argument("config_file")

    args = vars(parser.parse_args())

    config_file = expanded_path(args["config_file"])
    output_dir = expanded_path(args["output_dir"])
    sonic_method = args["sonic_method"]
    output_file = args.get("output_file", None)
    store_internal = args.get("store_internal", True)
    overrides = validate_overrides(args.get("override", []))

    with log_handler(args), redirected_warnings(), redirected_logging():
        print(solve(
            output_file=output_file, sonic_method=sonic_method,
            config_file=config_file, output_dir=output_dir,
            store_internal=store_internal, overrides=overrides,
        ))
