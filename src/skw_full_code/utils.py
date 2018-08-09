# -*- coding: utf-8 -*-
"""
Useful functions
"""

from enum import IntEnum
from math import pi
from configparser import ConfigParser
from pathlib import Path

import numpy as np
from numpy import sqrt

import logbook

from stringtopy import (
    str_to_float_converter, str_to_int_converter, str_to_bool_converter
)

logger = logbook.Logger(__name__)

str_to_float = str_to_float_converter()
str_to_int = str_to_int_converter()
str_to_bool = str_to_bool_converter()


class MHD_Wave_Index(IntEnum):
    """
    Enum for MHD wave speed indexes
    """
    slow = 0
    alfven = 1
    fast = 2


def is_supersonic(v, B, rho, sound_speed, mhd_wave_type):
    """
    Checks whether velocity is supersonic.
    """
    speeds = mhd_wave_speeds(B, rho, sound_speed)

    v_axis = 1 if v.ndim > 1 else 0
    v_sq = np.sum(v**2, axis=v_axis)

    with np.errstate(invalid="ignore"):
        return v_sq > speeds[MHD_Wave_Index[mhd_wave_type]]


def mhd_wave_speeds(B, rho, sound_speed):
    """
    Computes MHD wave speeds (slow, alfven, fast)
    """
    B_axis = 1 if B.ndim == 2 else 0

    B_sq = np.sum(B**2, axis=B_axis)

    if B_axis:
        cos_sq_psi = B[:, ODEIndex.B_θ]**2 / B_sq
    else:
        cos_sq_psi = B[ODEIndex.B_θ]**2 / B_sq

    v_a_sq = B_sq / (4*pi*rho)
    slow = 1/2 * (
        v_a_sq + sound_speed**2 - sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    alfven = v_a_sq * cos_sq_psi
    fast = 1/2 * (
        v_a_sq + sound_speed**2 + sqrt(
            (v_a_sq + sound_speed**2)**2 -
            4 * v_a_sq * sound_speed**2 * cos_sq_psi
        )
    )
    return slow, alfven, fast


def find_in_array(array, item):
    """
    Finds item in array or returns None
    """
    if array.ndim > 1:
        raise TypeError("array must be 1D")
    try:
        return list(array).index(item)
    except ValueError:
        return None


def cli_to_var(cmd):
    """
    Convert cli style argument to valid python name
    """
    return cmd.replace("-", "_")


def expanded_path(*path):
    """
    Return expanded pathlib.Path object
    """
    return Path(*path).expanduser().absolute()  # pylint: disable=no-member


class CaseDependentConfigParser(ConfigParser):
    # pylint: disable=too-many-ancestors
    """
    configparser.ConfigParser subclass that removes the case transform.
    """
    def optionxform(self, optionstr):
        return optionstr


def get_solutions(run, soln_range):
    """
    Get solutions based on range
    """
    if soln_range is None:
        soln_range = "0"
    elif soln_range == "final":
        return run.final_solution
    return run.solutions[soln_range]


class ODEIndex(IntEnum):
    """
    Enum for array index for variables in the odes
    """
    w_r = 0
    w_φ = 1
    w_Er = 2
    b_φ = 3
    b_r = 4
    ln_ρ = 5


MAGNETIC_INDEXES = [ODEIndex.b_r, ODEIndex.b_φ]
VELOCITY_INDEXES = [ODEIndex.w_r, ODEIndex.w_φ]


class SKWSolverError(Exception):
    """
    Base error class
    """
    pass
