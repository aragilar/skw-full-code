"""
Defines the dumpers for the data strutures
"""
from h5preserve import GroupContainer, OnDemandGroupContainer

from ._containers import (
    Solution, SolutionInput, ConfigInput, Run, InitialConditions, Solutions
)
from ._utils import ds_registry


# pylint: disable=missing-docstring

@ds_registry.dumper(InitialConditions, "InitialConditions", version=2)
def _initial_dump(initial_conditions):
    return GroupContainer(
        attrs={
            "a_0": initial_conditions.a_0,
            "σ_O_0": initial_conditions.σ_O_0,
            "σ_P_0": initial_conditions.σ_P_0,
            "σ_H_0": initial_conditions.σ_H_0,
            "ρ_s": initial_conditions.ρ_s,
            "init_con": initial_conditions.init_con,
        }, heights=initial_conditions.heights,
    )


@ds_registry.dumper(Solution, "Solution", version=1)
def _solution_dumper(solution):
    return GroupContainer(
        attrs={
            "flag": solution.flag,
        },
        heights=solution.heights,
        solution=solution.solution,
        initial_conditions=solution.initial_conditions,
        t_roots=solution.t_roots,
        y_roots=solution.y_roots,
        solution_input=solution.solution_input,
    )


@ds_registry.dumper(ConfigInput, "ConfigInput", version=2)
def _config_dumper(config_input):
    return GroupContainer(
        attrs={
            "start": config_input.start,
            "stop": config_input.stop,
            "max_steps": config_input.max_steps,
            "num_heights": config_input.num_heights,
            "label": config_input.label,
            "relative_tolerance": config_input.relative_tolerance,
            "absolute_tolerance": config_input.absolute_tolerance,
            "v_rin_on_c_s": config_input.v_rin_on_c_s,
            "v_a_on_c_s": config_input.v_a_on_c_s,
            "σ_O_0": config_input.σ_O_0,
            "σ_P_0": config_input.σ_P_0,
            "σ_H_0": config_input.σ_H_0,
            "ρ_s": config_input.ρ_s,
        },
    )


@ds_registry.dumper(SolutionInput, "SolutionInput", version=2)
def _input_dumper(solution_input):
    return GroupContainer(
        attrs={
            "start": solution_input.start,
            "stop": solution_input.stop,
            "max_steps": solution_input.max_steps,
            "num_heights": solution_input.num_heights,
            "relative_tolerance": solution_input.relative_tolerance,
            "absolute_tolerance": solution_input.absolute_tolerance,
            "v_rin_on_c_s": solution_input.v_rin_on_c_s,
            "v_a_on_c_s": solution_input.v_a_on_c_s,
            "σ_O_0": solution_input.σ_O_0,
            "σ_P_0": solution_input.σ_P_0,
            "σ_H_0": solution_input.σ_H_0,
            "ρ_s": solution_input.ρ_s,
        },
    )


@ds_registry.dumper(Run, "Run", version=1)
def _run_dumper(run):
    return GroupContainer(
        time=run.time,
        config_filename=run.config_filename,
        config_input=run.config_input,
        float_type=run.float_type,
        final_solution=run.final_solution,
        solutions=run.solutions,
        skw_full_code_version=run.skw_full_code_version,
    )


@ds_registry.dumper(Solutions, "Solutions", version=1)
def _solutions_dumper(solutions):
    return OnDemandGroupContainer(**solutions)

# pylint: enable=missing-docstring
