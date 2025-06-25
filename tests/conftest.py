from datetime import datetime
import os
import shutil
import tempfile
import sys

import pytest

# to avoid having to specify PYTHONPATH when running tests
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

def pytest_addoption(parser):
    parser.addoption(
        '--keep-tmp',
        action='store_true',
        default=False,
        help='Keep temporary test directories',
    )

@pytest.fixture
def keep_tmp_dir(request):
    original_dir = os.getcwd()
    tmp_dir = tempfile.mkdtemp(
        prefix=f'zener_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}_',
    )
    os.chdir(tmp_dir)

    yield tmp_dir

    os.chdir(original_dir)

    if not request.config.getoption('--keep-tmp'):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f'\nKept temporary directory: {tmp_dir}')
