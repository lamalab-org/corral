"""Corral CLI - Command line interface for running benchmarks."""

import typer
from rich.console import Console

from corral.cli.bench import bench_app

app = typer.Typer(
    name="corral",
    help="Corral - AI Agent Benchmarking Framework",
    no_args_is_help=True,
)

# Add subcommands
app.add_typer(bench_app, name="bench", help="Run benchmarks against environments")

console = Console()


@app.callback()
def main():
    """Corral CLI for running agent benchmarks."""


if __name__ == "__main__":
    app()
