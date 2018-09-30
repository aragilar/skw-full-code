"""
Defines the loaders for the data strutures
"""

from ._containers import (
    Solution, SolutionInput, ConfigInput, Run,
    InitialConditions, Solutions
)
from ._utils import ds_registry


# pylint: disable=missing-docstring


@ds_registry.loader("InitialConditions", version=1)
def _initial_load(group):
    return InitialConditions(
        a_0=group.attrs["a_0"],
        σ_O_0=group.attrs["σ_O_0"],
        σ_P_0=group.attrs["σ_P_0"],
        σ_H_0=group.attrs["σ_H_0"],
        ρ_s=group.attrs["ρ_s"],
        z_s=group.attrs["z_s"],
        init_con=group.attrs["init_con"],
        heights=group["heights"]["data"],
    )


@ds_registry.loader("InitialConditions", version=2)
def _initial_load2(group):
    return InitialConditions(
        a_0=group.attrs["a_0"],
        σ_O_0=group.attrs["σ_O_0"],
        σ_P_0=group.attrs["σ_P_0"],
        σ_H_0=group.attrs["σ_H_0"],
        ρ_s=group.attrs["ρ_s"],
        init_con=group.attrs["init_con"],
        heights=group["heights"]["data"],
    )


@ds_registry.loader("Solution", version=1)
def _solution_loader_(group):
    if group["t_roots"] is None:
        t_roots = None
    else:
        t_roots = group["t_roots"]["data"]
    if group["y_roots"] is None:
        y_roots = None
    else:
        y_roots = group["y_roots"]["data"]

    return Solution(
        flag=group.attrs["flag"],
        heights=group["heights"]["data"],
        solution=group["solution"]["data"],
        initial_conditions=group["initial_conditions"],
        solution_input=group["solution_input"],
        t_roots=t_roots,
        y_roots=y_roots,
    )


@ds_registry.loader("ConfigInput", version=1)
def _config_loader(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        max_steps=group.attrs["max_steps"],
        num_heights=group.attrs["num_heights"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        σ_O_0=group.attrs["σ_O_0"],
        σ_P_0=group.attrs["σ_P_0"],
        σ_H_0=group.attrs["σ_H_0"],
        ρ_s=group.attrs["ρ_s"],
        z_s=group.attrs["z_s"],
    )


@ds_registry.loader("ConfigInput", version=2)
def _config_loader2(group):
    return ConfigInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        max_steps=group.attrs["max_steps"],
        num_heights=group.attrs["num_heights"],
        label=group.attrs["label"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        σ_O_0=group.attrs["σ_O_0"],
        σ_P_0=group.attrs["σ_P_0"],
        σ_H_0=group.attrs["σ_H_0"],
        ρ_s=group.attrs["ρ_s"],
    )


@ds_registry.loader("SolutionInput", version=1)
def _input_loader(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        max_steps=group.attrs["max_steps"],
        num_heights=group.attrs["num_heights"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        σ_O_0=group.attrs["σ_O_0"],
        σ_P_0=group.attrs["σ_P_0"],
        σ_H_0=group.attrs["σ_H_0"],
        ρ_s=group.attrs["ρ_s"],
        z_s=group.attrs["z_s"],
    )


@ds_registry.loader("SolutionInput", version=2)
def _input_loader2(group):
    return SolutionInput(
        start=group.attrs["start"],
        stop=group.attrs["stop"],
        max_steps=group.attrs["max_steps"],
        num_heights=group.attrs["num_heights"],
        relative_tolerance=group.attrs["relative_tolerance"],
        absolute_tolerance=group.attrs["absolute_tolerance"],
        v_rin_on_c_s=group.attrs["v_rin_on_c_s"],
        v_a_on_c_s=group.attrs["v_a_on_c_s"],
        σ_O_0=group.attrs["σ_O_0"],
        σ_P_0=group.attrs["σ_P_0"],
        σ_H_0=group.attrs["σ_H_0"],
        ρ_s=group.attrs["ρ_s"],
    )


@ds_registry.loader("Run", version=1)
def _run_loader(group):
    return Run(
        config_input=group["config_input"],
        config_filename=group["config_filename"],
        time=group["time"],
        final_solution=group["final_solution"],
        solutions=group["solutions"],
        skw_full_code_version=group["skw_full_code_version"],
        float_type=group["float_type"],
    )


@ds_registry.loader("Solutions", version=1)
def _solutions_loader(group):
    return Solutions(**group)

# pylint: enable=missing-docstring
