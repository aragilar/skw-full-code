# -*- coding: utf-8 -*-
from pathlib import Path

from skw_full_code.solve import solve


class TestSolve:
    def test_default(self, tmpdir):
        solve(
            output_dir=Path(str(tmpdir)), config_file=None, output_file=None,
        )
