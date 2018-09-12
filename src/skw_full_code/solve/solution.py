# -*- coding: utf-8 -*-
"""
The system of odes
"""
import logbook
from numpy import sqrt, exp
from scikits.odes import ode
from scikits.odes.sundials import (
    CVODESolveFailed, CVODESolveFoundRoot, CVODESolveReachedTSTOP
)
from scikits.odes.sundials.cvode import StatusEnum

from .utils import (
    error_handler, SolverError,
)

from ..float_handling import float_type
from ..file_format import Solution
from ..utils import ODEIndex

INTEGRATOR = "cvode"
LINSOLVER = "dense"

log = logbook.Logger(__name__)


def ode_system(*, a_0, σ_O_0, σ_P_0, σ_H_0, ρ_s, z_s):
    """
    Set up the system we are solving for.
    """
    def rhs_func(z, variables, derivs):
        """
        Compute the ODEs
        """
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


def solution(
    soln_input, initial_conditions, *, onroot_func=None, tstop=None,
    ontstop_func=None, root_func=None, root_func_args=None
):
    """
    Find solution
    """
    heights = initial_conditions.heights
    init_con = initial_conditions.init_con
    a_0 = initial_conditions.a_0
    σ_O_0 = initial_conditions.σ_O_0
    σ_P_0 = initial_conditions.σ_P_0
    σ_H_0 = initial_conditions.σ_H_0
    ρ_s = initial_conditions.ρ_s
    z_s = initial_conditions.z_s
    absolute_tolerance = soln_input.absolute_tolerance
    relative_tolerance = soln_input.relative_tolerance
    max_steps = soln_input.max_steps

    soln = main_solution(
        heights=heights, initial_conditions=init_con, a_0=a_0, σ_O_0=σ_O_0,
        σ_P_0=σ_P_0, σ_H_0=σ_H_0, ρ_s=ρ_s, z_s=z_s,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance, max_steps=max_steps,
        onroot_func=onroot_func, tstop=tstop, ontstop_func=ontstop_func,
        root_func=root_func, root_func_args=root_func_args
    )

    return Solution(
        solution_input=soln_input, initial_conditions=initial_conditions,
        flag=soln.flag, heights=soln.values.t, solution=soln.values.y,
        t_roots=soln.roots.t, y_roots=soln.roots.y,
    )
