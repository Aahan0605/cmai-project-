import pytest

from optlab.cli import build_parser


@pytest.mark.parametrize("command", ["tune", "train", "sweep", "cv"])
def test_experiment_commands_require_a_config_selection(command):
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


def test_experiment_name_and_config_cannot_be_ambiguous():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["sweep", "--experiment", "a", "--config", "b.yaml"])
