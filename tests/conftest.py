from datetime import datetime
import shutil
import tempfile

import pytest

def pytest_addoption(parser):
    parser.addoption(
        '--keep-tmp',
        action='store_true',
        default=False,
        help='Keep temporary test directories',
    )

@pytest.fixture
def keep_tmp_dir(request):
    tmp_dir = tempfile.mkdtemp(
        prefix=f'zener_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}_',
    )
    yield tmp_dir

    if not request.config.getoption('--keep-tmp'):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        print(f'\nKept temporary directory: {tmp_dir}')
