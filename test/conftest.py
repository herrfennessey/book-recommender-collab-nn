import os
import pathlib

cwd = pathlib.Path(__file__).parent
os.environ["MODEL_FOLDER"] = (cwd / "files").as_posix()

pytest_plugins = [
    "test.fixtures.test_client",
]
