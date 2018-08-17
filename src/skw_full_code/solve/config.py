# -*- coding: utf-8 -*-
"""
Define input and environment for ode system
"""

import logbook

import numpy as np

from ..float_handling import float_type, FLOAT_TYPE
from ..file_format import ConfigInput, InitialConditions, SolutionInput
from ..utils import (
    str_to_float, str_to_int, CaseDependentConfigParser, ODEIndex,
)

from .utils import SolverError, add_overrides

log = logbook.Logger(__name__)


def define_conditions(inp):
    """
    Compute initial conditions based on input
    """
    init_con = np.zeros(len(ODEIndex), dtype=FLOAT_TYPE)
    ε = inp.v_rin_on_c_s
    a_0 = inp.v_a_on_c_s
    σ_O_0 = inp.σ_O_0
    σ_P_0 = inp.σ_P_0
    σ_H_0 = inp.σ_H_0
    ρ_s = inp.ρ_s
    z_s = inp.z_s

    σ_perp_sq = σ_P_0 ** 2 + σ_H_0 ** 2
    w_φ = - σ_H_0 / σ_P_0 * ε / 4 - a_0 ** 2 * σ_perp_sq * ε / (2 * σ_P_0)
    w_Er = σ_H_0 / σ_P_0 * ε + ε / (2 * a_0 ** 2 * σ_P_0) - w_φ

    init_con[ODEIndex.w_r] = - ε
    init_con[ODEIndex.w_φ] = w_φ
    init_con[ODEIndex.w_Er] = w_Er
    init_con[ODEIndex.ln_ρ] = 1
    init_con[ODEIndex.b_r] = 0
    init_con[ODEIndex.b_φ] = 0

    heights = np.linspace(inp.start, inp.stop, inp.num_heights)
    if np.any(np.isnan(init_con)):
        raise SolverError("Input implies NaN")

    return InitialConditions(
        heights=heights, init_con=init_con, a_0=a_0, σ_O_0=σ_O_0,
        σ_P_0=σ_P_0, σ_H_0=σ_H_0, ρ_s=ρ_s, z_s=z_s,
    )


def get_input_from_conffile(*, config_file, overrides=None):
    """
    Get input values
    """
    config = CaseDependentConfigParser()
    if config_file:
        with config_file.open("r") as f:
            config.read_file(f)

    return add_overrides(overrides=overrides, config_input=ConfigInput(
        start=config.get("config", "start", fallback="0"),
        stop=config.get("config", "stop", fallback="5"),
        max_steps=config.get("config", "max_steps", fallback="10000"),
        num_heights=config.get("config", "num_heights", fallback="10000"),
        label=config.get("config", "label", fallback="default"),
        relative_tolerance=config.get(
            "config", "relative_tolerance", fallback="1e-6"
        ),
        absolute_tolerance=config.get(
            "config", "absolute_tolerance", fallback="1e-10"
        ),
        v_rin_on_c_s=config.get("initial", "v_rin_on_c_s", fallback="1"),
        v_a_on_c_s=config.get("initial", "v_a_on_c_s", fallback="1"),
        σ_O_0=config.get("initial", "σ_O_0", fallback="100"),
        σ_P_0=config.get("initial", "σ_P_0", fallback="3"),
        σ_H_0=config.get("initial", "σ_H_0", fallback="4"),
        ρ_s=config.get("initial", "ρ_s", fallback="1e-6"),
        z_s=config.get("initial", "z_s", fallback="40"),
    ))


def config_input_to_soln_input(inp):
    """
    Convert user input into solver input
    """
    return SolutionInput(
        start=float_type(str_to_float(inp.start)),
        stop=float_type(str_to_float(inp.stop)),
        max_steps=str_to_int(inp.max_steps),
        num_heights=str_to_int(inp.num_heights),
        relative_tolerance=float_type(str_to_float(inp.relative_tolerance)),
        absolute_tolerance=float_type(str_to_float(inp.absolute_tolerance)),
        v_rin_on_c_s=float_type(str_to_float(inp.v_rin_on_c_s)),
        v_a_on_c_s=float_type(str_to_float(inp.v_a_on_c_s)),
        σ_O_0=float_type(str_to_float(inp.σ_O_0)),
        σ_P_0=float_type(str_to_float(inp.σ_P_0)),
        σ_H_0=float_type(str_to_float(inp.σ_H_0)),
        ρ_s=float_type(str_to_float(inp.ρ_s)),
        z_s=float_type(str_to_float(inp.z_s)),
    )
