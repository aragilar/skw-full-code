"""
Defines the common data structures
"""

from collections import defaultdict
from collections.abc import MutableMapping

import attr
from numpy import asarray, concatenate, zeros

from h5preserve import wrap_on_demand, OnDemandWrapper, DelayedContainer

from ..utils import ODEIndex


# pylint: disable=too-few-public-methods

@attr.s(cmp=False, hash=False)
class Solution:
    """
    Container for result from solver
    """
    solution_input = attr.ib()
    angles = attr.ib()
    solution = attr.ib()
    flag = attr.ib()
    coordinate_system = attr.ib()
    initial_conditions = attr.ib()
    t_roots = attr.ib()
    y_roots = attr.ib()


@attr.s
class ConfigInput:
    """
    Container for input from config file
    """
    start = attr.ib()
    stop = attr.ib()
    max_steps = attr.ib()
    num_angles = attr.ib()
    label = attr.ib()
    relative_tolerance = attr.ib()
    absolute_tolerance = attr.ib()
    target_velocity = attr.ib()
    split_method = attr.ib()
    v_rin_on_c_s = attr.ib()
    v_a_on_c_s = attr.ib()
    c_s_on_v_k = attr.ib()
    η_O = attr.ib()
    η_H = attr.ib()
    η_A = attr.ib()


@attr.s
class SolutionInput:
    """
    Container for parsed input for solution
    """


class Solutions(MutableMapping):
    """
    Container holding the different solutions generated
    """
    def __init__(self, **solutions):
        self._solutions = {}
        self.update(solutions)

    def __getitem__(self, key):
        value = self._solutions[key]
        if isinstance(value, OnDemandWrapper):
            value = value()
            self._solutions[key] = value
        return value

    def __setitem__(self, key, val):
        self._solutions[key] = wrap_on_demand(self, key, val)

    def __delitem__(self, key):
        del self._solutions[key]

    def __iter__(self):
        for key in self._solutions:
            yield key

    def __len__(self):
        return len(self._solutions)

    def _h5preserve_update(self):
        """
        Support for h5preserve on demand use
        """
        for key, val in self.items():
            self._solutions[key] = wrap_on_demand(self, key, val)

    def __repr__(self):
        return "Solutions(" + ', '.join(
            "{key}={val}".format(key=key, val=val)
            for key, val in self._solutions.items()
        ) + ")"

    def add_solution(self, soln):
        """
        Add a solution returning the index of the solution
        """
        index = self._get_next_index()
        self[index] = soln
        return index

    def _get_next_index(self):
        """
        Get the next available index
        """
        if str(len(self._solutions)) not in self._solutions:
            return str(len(self._solutions))
        else:
            raise RuntimeError("Failed to guess a solution location")


@attr.s(cmp=False, hash=False)
class Run:
    """
    Container holding a single run of the solver code
    """
    config_input = attr.ib()
    config_filename = attr.ib()
    skw_full_code_version = attr.ib()
    float_type = attr.ib()
    sonic_method = attr.ib()
    time = attr.ib(default=None)
    _final_solution = attr.ib(default=attr.Factory(DelayedContainer))
    solutions = attr.ib(default=attr.Factory(Solutions))

    @property
    def final_solution(self):
        """
        The best solution found
        """
        return self._final_solution

    @final_solution.setter
    def final_solution(self, soln):
        if isinstance(self._final_solution, DelayedContainer):
            self._final_solution.write_container(soln)
            self._final_solution = soln
        else:
            raise RuntimeError("Cannot change final solution")


@attr.s(cmp=False, hash=False)
class InitialConditions:
    """
    Container holding the initial conditions for the solver
    """
    pass
# pylint: enable=too-few-public-methods
