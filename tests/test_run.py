# -*- coding: utf-8 -*-
from pathlib import Path

from skw_full_code.solve import solve
from skw_full_code.analyse.info import info


class TestSolve:
    def test_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), config_file=None, output_file=None,
        )


class TestAnalysis:
    def test_info_run(self, solution, tmp_text_stream):
        info(
            solution, group="run", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_status(self, solution, tmp_text_stream):
        info(
            solution, group="status", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_input(self, solution, tmp_text_stream):
        info(
            solution, group="input", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_initial_conditions(self, solution, tmp_text_stream):
        info(
            solution, group="initial-conditions", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_sonic_points(self, solution, tmp_text_stream):
        info(
            solution, group="sonic-points", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_crosses_points(self, solution, tmp_text_stream):
        info(
            solution, group="crosses-points", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_info_solutions(self, solution, tmp_text_stream):
        info(
            solution, group="solutions", soln_range=None,
            output_file=tmp_text_stream,
        )
