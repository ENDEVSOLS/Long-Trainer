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


# ─── bot management ───────────────────────────────────────────────────────────

def _get_trainer(config_path: str = "longtrainer.yaml"):
    import os
    import yaml
    from longtrainer.trainer import LongTrainer

    if not os.path.exists(config_path):
        click.secho(f"❌ Config file '{config_path}' not found. Run 'longtrainer init' first.", fg="red", bold=True)
        raise SystemExit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return LongTrainer(
        mongo_endpoint=cfg.get("mongo_endpoint", "mongodb://localhost:27017/"),
        chunk_size=cfg.get("chunking", {}).get("chunk_size", 2048),
        chunk_overlap=cfg.get("chunking", {}).get("chunk_overlap", 200),
        encrypt_chats=cfg.get("encrypt_chats", False),
    )


@cli.group()
def bot() -> None:
    """Manage LongTrainer bots."""
    pass


@bot.command("list")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def bot_list(config: str) -> None:
    """List all deployed bots."""
    trainer = _get_trainer(config)
    bots = list(trainer.bots.find({}, {"_id": 0}))
    if not bots:
        click.secho("No bots found.", fg="yellow")
        return

    click.secho(f"\n🤖 Found {len(bots)} bots:", fg="cyan", bold=True)
    for b in bots:
        click.echo(f"  - {b.get('bot_id', 'Unknown')}")


@bot.command("create")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
@click.option("--prompt", "-p", default=None, help="Custom system prompt.")
def bot_create(config: str, prompt: str | None) -> None:
    """Initialize a new empty bot."""
    trainer = _get_trainer(config)
    bot_id = trainer.initialize_bot_id()
    if not bot_id:
        click.secho("❌ Failed to create bot.", fg="red", bold=True)
        raise SystemExit(1)

    trainer.create_bot(bot_id, prompt_template=prompt)
    click.secho(f"✅ Created new bot: {bot_id}", fg="green", bold=True)


@bot.command("delete")
@click.argument("bot_id")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def bot_delete(bot_id: str, config: str) -> None:
    """Delete a bot and all its data."""
    trainer = _get_trainer(config)
    try:
        trainer.delete_chatbot(bot_id)
        click.secho(f"🗑️ Deleted bot: {bot_id}", fg="yellow")
    except ValueError as e:
        click.secho(f"❌ Error: {e}", fg="red")
        raise SystemExit(1)


# ─── documents ────────────────────────────────────────────────────────────────

@cli.command("add-doc")
@click.argument("bot_id")
@click.argument("path")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def add_doc(bot_id: str, path: str, config: str) -> None:
    """Upload a document to a bot. PATH can be a file path or URL."""
    trainer = _get_trainer(config)

    try:
        trainer.load_bot(bot_id)
    except Exception as e:
        click.secho(f"❌ Failed to load bot {bot_id}.", fg="red")
        raise SystemExit(1)

    click.echo(f"⏳ Ingesting '{path}' into {bot_id} ...")

    if path.startswith("http://") or path.startswith("https://"):
        trainer.add_document_from_link([path], bot_id)
    else:
        import os
        if not os.path.exists(path):
            click.secho(f"❌ File '{path}' does not exist.", fg="red")
            raise SystemExit(1)
        trainer.add_document_from_path(path, bot_id)

    # Refresh the FAISS index by creating the bot again
    trainer.create_bot(bot_id)
    click.secho("✅ Document added successfully!", fg="green", bold=True)


# ─── chat ─────────────────────────────────────────────────────────────────────

@cli.command("chat")
@click.argument("bot_id")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def chat_command(bot_id: str, config: str) -> None:
    """Start an interactive terminal chat with a bot."""
    trainer = _get_trainer(config)

    click.echo(f"⏳ Loading bot {bot_id} ...")
    try:
        trainer.load_bot(bot_id)
    except Exception as e:
        click.secho(f"❌ Failed to load bot {bot_id}: {e}", fg="red")
        raise SystemExit(1)

    chat_id = trainer.new_chat(bot_id)

    click.secho(f"\n💬 Chat session started (ID: {chat_id})", fg="cyan", bold=True)
    click.secho("Type 'exit' or 'quit' to end the session.\n", fg="cyan")

    while True:
        try:
            query = click.prompt(click.style("You", fg="green", bold=True))
            if query.lower() in ("exit", "quit"):
                break

            click.secho("Bot: ", fg="blue", bold=True, nl=False)

            for chunk in trainer.get_response(query, bot_id, chat_id, stream=True):
                click.echo(chunk, nl=False)
            click.echo()  # Newline after response

        except (KeyboardInterrupt, EOFError):
            click.echo()
            break

    click.secho("👋 Session ended.", fg="yellow")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
