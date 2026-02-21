"""LongTrainer CLI — command-line interface.

Usage:
    longtrainer --version
    longtrainer init
    longtrainer serve
"""

from __future__ import annotations

import click

from longtrainer import __version__


@click.group()
@click.version_option(__version__, prog_name="longtrainer")
def cli() -> None:
    """LongTrainer — Production-Ready RAG Framework CLI."""


# ─── init ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--mongo",
    prompt="MongoDB URI",
    default="mongodb://localhost:27017/",
    help="MongoDB connection string.",
)
@click.option(
    "--llm-provider",
    prompt="LLM provider",
    type=click.Choice(["openai", "ollama", "anthropic"], case_sensitive=False),
    default="openai",
    help="LLM provider to use.",
)
@click.option(
    "--model-name",
    prompt="Model name",
    default="gpt-4o-2024-08-06",
    help="Model name for the chosen provider.",
)
@click.option(
    "--embedding-model",
    prompt="Embedding model",
    default="text-embedding-3-small",
    help="Embedding model name.",
)
@click.option(
    "--chunk-size",
    prompt="Chunk size",
    default=2048,
    type=int,
    help="Text splitter chunk size.",
)
@click.option(
    "--chunk-overlap",
    prompt="Chunk overlap",
    default=200,
    type=int,
    help="Text splitter overlap.",
)
@click.option(
    "--encrypt-chats",
    is_flag=True,
    default=False,
    help="Enable Fernet encryption for stored chats.",
)
@click.option(
    "--output",
    "-o",
    default="longtrainer.yaml",
    help="Output config file path.",
)
def init(
    mongo: str,
    llm_provider: str,
    model_name: str,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
    encrypt_chats: bool,
    output: str,
) -> None:
    """Initialize a new LongTrainer project with a config file."""
    import yaml

    config = {
        "mongo_endpoint": mongo,
        "llm": {
            "provider": llm_provider,
            "model_name": model_name,
        },
        "embedding": {
            "model_name": embedding_model,
        },
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "encrypt_chats": encrypt_chats,
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }

    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.secho(f"\n✅ Config written to {output}", fg="green", bold=True)
    click.echo("  Next steps:")
    click.echo(f"    1. Review and edit {output}")
    click.echo("    2. Run: longtrainer serve")


# ─── serve ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option(
    "--config",
    "-c",
    default="longtrainer.yaml",
    type=click.Path(exists=True),
    help="Path to longtrainer.yaml config file.",
)
@click.option("--host", default=None, help="Override host (default: from config).")
@click.option("--port", "-p", default=None, type=int, help="Override port (default: from config).")
@click.option("--reload", "reload_", is_flag=True, default=False, help="Enable auto-reload for development.")
def serve(config: str, host: str | None, port: int | None, reload_: bool) -> None:
    """Start the LongTrainer API server."""
    import yaml

    try:
        import uvicorn
    except ImportError:
        click.secho(
            "❌ uvicorn not installed. Run: pip install longtrainer[api]",
            fg="red",
            bold=True,
        )
        raise SystemExit(1)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    server_cfg = cfg.get("server", {})
    final_host = host or server_cfg.get("host", "0.0.0.0")
    final_port = port or server_cfg.get("port", 8000)

    click.secho(
        f"\n🚀 Starting LongTrainer API server on {final_host}:{final_port}",
        fg="cyan",
        bold=True,
    )

    uvicorn.run(
        "longtrainer.api:app",
        host=final_host,
        port=final_port,
        reload=reload_,
    )


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
