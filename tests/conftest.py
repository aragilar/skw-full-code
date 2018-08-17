from pathlib import Path

import pytest

from skw_full_code.solve import solve
from skw_full_code.float_handling import float_type as FLOAT_TYPE

PLOT_FILE = "plot.png"


@pytest.fixture(scope="session")
def solution_default(tmpdir):
    return solve(
        output_dir=Path(str(tmpdir)), output_file=None, config_file=None,
    )


@pytest.fixture()
def mpl_interactive():
    import matplotlib.pyplot as plt
    plt.ion()


@pytest.fixture
def tmp_text_stream(request):
    from io import StringIO
    stream = StringIO()

    def fin():
        stream.close()

    request.addfinalizer(fin)
    return stream


@pytest.fixture
def plot_file(tmpdir):
    return Path(Path(str(tmpdir)), PLOT_FILE)


@pytest.fixture
def test_id(request):
    return str(request.node) + str(FLOAT_TYPE)
