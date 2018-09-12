# -*- coding: utf-8 -*-
from pathlib import Path

import pytest

from skw_full_code.solve import solve
from skw_full_code.analyse.info import info
from skw_full_code.analyse.plot import plot


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

    def test_info_solutions(self, solution, tmp_text_stream):
        info(
            solution, group="solutions", soln_range=None,
            output_file=tmp_text_stream,
        )

    def test_plot_show(self, solution, mpl_interactive):
        plot(solution, show=True)

    @pytest.mark.mpl_image_compare
    def test_plot_file(self, solution, plot_file):
        return plot(solution, plot_filename=plot_file, close=False)
