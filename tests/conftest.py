import sys
import os
import pytest

# Insert root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.database import init_db
from app.utils.model_loader import model_loader

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    init_db()
    model_loader.load_artifacts()
