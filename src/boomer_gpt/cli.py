"""Command line interface for boomer-gpt."""
import logging

import click

from boomer_gpt import __version__
from boomer_gpt.main import demo, BoomerGPT

__all__ = [
    "main",
]

logger = logging.getLogger(__name__)


@click.group()
@click.option("-v", "--verbose", count=True)
@click.option("-q", "--quiet")
@click.version_option(__version__)
def main(verbose: int, quiet: bool):
    """CLI for boomer-gpt.

    :param verbose: Verbosity while running.
    :param quiet: Boolean to be quiet or verbose.
    """
    if verbose >= 2:
        logger.setLevel(level=logging.DEBUG)
    elif verbose == 1:
        logger.setLevel(level=logging.INFO)
    else:
        logger.setLevel(level=logging.WARNING)
    if quiet:
        logger.setLevel(level=logging.ERROR)


@main.command()
@click.argument("config")
def run(config: str):
    """Run the boomer-gpt's demo command."""
    engine = BoomerGPT()
    engine.load_configuration(config)
    engine.run()


if __name__ == "__main__":
    main()
