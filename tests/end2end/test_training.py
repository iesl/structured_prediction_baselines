from typing import Dict
from pathlib import Path
from textwrap import dedent
import os
import pytest
from pytest_console_scripts import ScriptRunner
from .common import (
    config_directory,
    vocab_directory,
    root_directory,
    configs,
    marks,
)


base_training_command_template = """
allennlp
train
{config}
--include-package=structured_prediction_baselines
--overrides={{ "type": "default", "dataset_reader.max_instances": 100, "trainer.cuda_device": -1, "vocabulary": {{"type": "from_files", "directory": "{vocab_directory}" }} }}
-s {serialization_dir}
        """
# extra { } in overrides to escape format() call


@pytest.fixture(scope="session")
def training_env() -> Dict[str, str]:
    return {
        "DATA_DIR": str(Path(root_directory / "data")),
        "CUDA_DEVICE": "-1",
        "TEST": "1",
    }


params = [
    pytest.param(config, marks=pytest.mark.__getattr__(config.stem))
    for config in configs
]


@pytest.fixture(scope="session")
def model_directories(tmpdir_factory: pytest.TempPathFactory) -> Path:
    d = tmpdir_factory.mktemp("models")
    return d


@pytest.mark.training
@pytest.mark.parametrize(("config"), params)
def test_complete_training(
    config: Path,
    training_env: Dict[str, str],
    model_directories: Path,
) -> None:  # fixture
    os.environ.update(**training_env)
    script_runner = ScriptRunner("subprocess", config_directory)
    command = (
        base_training_command_template.format(
            config=config,
            serialization_dir=model_directories / config.stem,
            vocab_directory=vocab_directory,
        )
        .strip()
        .split("\n")
    )
    ret = script_runner.run(*command)
    assert ret.success
