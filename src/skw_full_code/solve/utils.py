# -*- coding: utf-8 -*-
"""
Utility function and classes for solver associated code
"""

import attr

from .. import __version__ as ds_version
from ..file_format import CONFIG_FIELDS
from ..logging import logging_options
from ..utils import SKWError


def error_handler(error_code, module, func, msg, user_data):
    """ drop all CVODE/IDA messages """
    # pylint: disable=unused-argument
    pass


def onroot_continue(*args):
    """
    Always continue after finding root
    """
    # pylint: disable=unused-argument
    return 0


def onroot_stop(*args):
    """
    Always stop after finding root
    """
    # pylint: disable=unused-argument
    return 1


def ontstop_continue(*args):
    """
    Always continue after finding tstop
    """
    # pylint: disable=unused-argument
    return 0


def ontstop_stop(*args):
    """
    Always stop after finding tstop
    """
    # pylint: disable=unused-argument
    return 1


def add_solver_arguments(parser):
    """
    Add common parser arguments for solver
    """
    parser.add_argument(
        '--version', action='version', version='%(prog)s ' + ds_version
    )
    parser.add_argument(
        "--sonic-method", choices=(
            "step", "single", "dae_single", "mcmc", "sonic_root",
            "hydrostatic",
        ), default="single",
    )
    parser.add_argument("--output-file")
    parser.add_argument(
        "--output-dir", default=".",
    )
    internal_store_group = parser.add_mutually_exclusive_group()
    internal_store_group.add_argument(
        "--store-internal", action='store_true', default=True
    )
    internal_store_group.add_argument(
        "--no-store-internal", action='store_false', dest="store_internal",
    )
    parser.add_argument("--override", action='append', nargs=2, default=[])
    logging_options(parser)


class SolverError(SKWError):
    """
    Error class for problems with solver routines
    """
    pass


def add_overrides(*, overrides, config_input):
    """
    Create new instance of `ConfigInput` with `overrides` added.
    """
    if overrides is None:
        return config_input
    return attr.assoc(config_input, **overrides)


def validate_overrides(overrides):
    """
    Validate overrides passed as auguments
    """
    clean_overrides = {}
    for name, value in overrides:
        if name in CONFIG_FIELDS:
            clean_overrides[name] = value
        else:
            raise SolverError("Override incorrect: no such field {}".format(
                name
            ))
    return clean_overrides
