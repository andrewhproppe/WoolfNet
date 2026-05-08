import logging

import click
import mlflow

from woolfnet.data.tasks import COMMANDS as data_commands
from woolfnet.gpt.tasks import COMMANDS as gpt_commands
from woolfnet.paths import MLFLOW_LOCAL_URI
from woolfnet.tokenization.tasks import COMMANDS as tokenizer_commands
from woolfnet.utils.general import RuntimeStatistics

COMMAND_MAP = {
    "data": data_commands,
    "tokenizer": tokenizer_commands,
    "gpt": gpt_commands,
}


@click.group()
@click.option(
    "--debug-level",
    type=click.Choice(choices=["INFO", "DEBUG", "WARNING"]),
    default="INFO",
)
@click.option(
    "--profile", is_flag=True, help="Run simple profiling during program runtime"
)
def cli(debug_level: str, profile: bool) -> None:
    """Main cli entrypoint"""
    logging.basicConfig(
        level=debug_level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="[%Y-%m-%d %H:%M:%S]",
    )
    if profile:
        RuntimeStatistics()
    mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)


@cli.group()
def data():
    """Main CLI group for data and datasets"""
    pass


for command in COMMAND_MAP["data"].values():
    data.add_command(command)


@cli.group()
def tokenizer():
    """Main CLI group for tokenizer training and testing"""
    pass


for command in COMMAND_MAP["tokenizer"].values():
    tokenizer.add_command(command)


@cli.group()
def gpt():
    """Main CLI group for GPT training and testing"""
    mlflow.set_experiment("gpt")
    pass


for command in COMMAND_MAP["gpt"].values():
    gpt.add_command(command)
