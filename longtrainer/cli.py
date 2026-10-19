"""LongTrainer CLI — command-line interface.

Usage:
    longtrainer --version
    longtrainer init
    longtrainer serve
"""

from __future__ import annotations

import click

from longtrainer import __version__


# ─── init ─────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(__version__, prog_name="longtrainer")
def cli() -> None:
    """LongTrainer — Production-Ready RAG Framework CLI."""


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
    "--embedding-provider",
    prompt="Embedding provider",
    type=click.Choice(["openai", "ollama", "huggingface", "cohere"], case_sensitive=False),
    default="openai",
    help="Embedding provider to use.",
)
@click.option(
    "--embedding-model",
    prompt="Embedding model",
    default="text-embedding-3-small",
    help="Embedding model name.",
)
@click.option(
    "--vectorstore-provider",
    prompt="Vector Store provider",
    default="faiss",
    type=click.Choice([
        "faiss", "pinecone", "chroma", "qdrant",
        "pgvector", "mongodb", "milvus", "weaviate", "elasticsearch"
    ]),
    help="Provider for vector database.",
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
    embedding_provider: str,
    embedding_model: str,
    vectorstore_provider: str,
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
            "provider": embedding_provider,
            "model_name": embedding_model,
        },
        "vector_store": {
            "provider": vectorstore_provider,
        },
        "chunking": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "encrypt_chats": encrypt_chats,
        "rate_limiting": {
            "enabled": False,
            "llm_rpm": 60,
            "embedding_rpm": 120,
            "tool_rpm": 30,
            "ingestion_rpm": 10,
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
        },
    }

    with open(output, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    click.secho(f"\n✅ Config written to {output}", fg="green", bold=True)
    click.echo("\nNext steps:")
    click.echo(f"  1. Review and edit {output}")
    click.echo("  2. Run: longtrainer serve\n")


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
        click.secho("❌ uvicorn not installed. Run: pip install longtrainer[api]", fg="red")
        raise SystemExit(1)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    server_cfg = cfg.get("server", {})
    final_host = host or server_cfg.get("host", "0.0.0.0")
    final_port = port or server_cfg.get("port", 8000)

    click.secho(f"\n🚀 LongTrainer {__version__} API Server", fg="cyan", bold=True)
    click.echo(f"   Host   : {final_host}")
    click.echo(f"   Port   : {final_port}")
    click.echo(f"   Docs   : http://{final_host}:{final_port}/docs")
    click.echo(f"   Config : {config}\n")

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
        click.secho(f"❌ Config file {config_path!r} not found.", fg="red")
        click.echo("Run: longtrainer init")
        raise SystemExit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    return LongTrainer(
        mongo_endpoint=cfg.get("mongo_endpoint", "mongodb://localhost:27017/"),
        llm_provider=cfg.get("llm", {}).get("provider", "openai"),
        default_llm=cfg.get("llm", {}).get("model_name", "gpt-4o-2024-08-06"),
        embedding_provider=cfg.get("embedding", {}).get("provider", "openai"),
        embedding_model_name=cfg.get("embedding", {}).get("model_name", "text-embedding-3-small"),
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
        click.secho("No bots deployed yet.", fg="yellow")
        click.echo("Create one with: longtrainer bot create")
        return

    click.secho(f"\n🤖 Deployed Bots ({len(bots)}):\n", fg="cyan", bold=True)
    for b in bots:
        agent = "Agent" if b.get("agent_mode") else "RAG"
        click.echo(f"  {b.get('bot_id', 'Unknown')}  [{agent}]  {b.get('db_path', '')}")
    click.echo()


def _validate_agent_type(agent_type: str | None) -> None:
    """Validate a purpose before connecting to services or allocating a bot."""
    if agent_type is not None:
        from longtrainer.agent_types import AgentTypeRegistry
        try:
            AgentTypeRegistry.get(agent_type)
        except ValueError as error:
            raise click.BadParameter(str(error), param_hint="--agent-type") from error


def _parse_tools(tools: str | None) -> list[str] | None:
    """Distinguish an omitted tool option from an explicit empty replacement."""
    return None if tools is None else [name.strip() for name in tools.split(",") if name.strip()]


@cli.command("build")
@click.argument("bot_id", required=False)
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to infrastructure config file.")
@click.option("--agent-type", default=None, help="Purpose: research, coding, financial, customer_support.")
@click.option("--prompt", "-p", default=None, help="Custom system prompt.")
@click.option("--agent", is_flag=True, help="Enable legacy agent mode when no purpose is selected.")
@click.option("--tools", "-t", default=None, help="Comma-separated tools; pass an empty string for no tools.")
def build(bot_id: str | None, config: str, agent_type: str | None,
          prompt: str | None, agent: bool, tools: str | None) -> None:
    """Build an existing BOT_ID, or initialize and build a new bot if omitted."""
    _validate_agent_type(agent_type)
    trainer = _get_trainer(config)
    try:
        if bot_id:
            trainer.load_bot(bot_id)
            if bot_id not in trainer.bot_data:
                raise ValueError(f"Failed to load bot {bot_id}")
            if agent_type is None and not agent and tools is None:
                trainer._rebuild_bot(bot_id, prompt_template=prompt)
                click.echo(f"Bot built: {bot_id}")
                return
            if agent_type is None and not agent:
                current = trainer.bot_data[bot_id]
                agent_type = current.get("agent_type")
                if agent_type and prompt is None:
                    prompt = current.get("prompt_template")
        else:
            bot_id = trainer.initialize_bot_id()
            if not bot_id:
                raise ValueError("Failed to initialize bot")
        trainer.create_bot(
            bot_id=bot_id, prompt_template=prompt, agent_mode=agent, tools=_parse_tools(tools),
            **({"agent_type": agent_type} if agent_type is not None else {}),
        )
        click.echo(f"Bot built: {bot_id}")
    except Exception as error:
        raise click.ClickException(str(error)) from error


@bot.command("create")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
@click.option("--prompt", "-p", default=None, help="Custom system prompt.")
@click.option("--agent", is_flag=True, default=False, help="Enable Agent mode with tool calling.")
@click.option("--tools", "-t", default=None, help="Comma-separated list of tools (e.g. 'wikipedia,arxiv').")
@click.option("--agent-type", default=None, help="Purpose: research, coding, financial, customer_support.")
def bot_create(config: str, prompt: str | None, agent: bool, tools: str | None, agent_type: str | None) -> None:
    """Initialize a new empty bot."""
    _validate_agent_type(agent_type)
    trainer = _get_trainer(config)
    bot_id = trainer.initialize_bot_id()
    if not bot_id:
        click.secho("❌ Failed to create bot.", fg="red")
        raise SystemExit(1)

    tool_list = _parse_tools(tools)
    try:
        trainer.create_bot(
            bot_id=bot_id, prompt_template=prompt, agent_mode=agent,
            tools=tool_list if agent_type is not None else (tool_list or None),
            **({"agent_type": agent_type} if agent_type is not None else {}),
        )
    except Exception as error:
        raise click.ClickException(str(error)) from error
    click.secho("\n✅ Bot created successfully!", fg="green", bold=True)
    click.echo(f"   Bot ID    : {bot_id}")
    if agent_type:
        click.echo(f"   Agent Type: {agent_type}")
    else:
        click.echo(f"   Agent Mode: {'Yes' if agent else 'No (RAG)'}")
    tool_summary = "purpose defaults" if agent_type and tool_list is None else (", ".join(tool_list or []) or "none")
    click.echo(f"   Tools     : {tool_summary}")
    click.echo("\nNext steps:")
    click.echo(f"  longtrainer add-doc {bot_id} <file.pdf>")
    click.echo(f"  longtrainer chat {bot_id}\n")


@bot.command("delete")
@click.argument("bot_id")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def bot_delete(bot_id: str, config: str) -> None:
    """Delete a bot and all its data."""
    trainer = _get_trainer(config)
    try:
        trainer.delete_chatbot(bot_id)
        click.secho(f"🗑️  Bot deleted: {bot_id}", fg="yellow")
        click.echo("All associated data, chat history, and vector store removed.")
    except ValueError as e:
        click.secho(f"❌ {e}", fg="red")
        raise SystemExit(1)


# ─── documents ────────────────────────────────────────────────────────────────

@cli.command("add-doc")
@click.argument("bot_id")
@click.argument("path")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def add_doc(bot_id: str, path: str, config: str) -> None:
    """Upload a document to a bot. PATH can be a file path or URL."""
    import time as _time
    from longtrainer.rate_limiter import LongTrainerRateLimitError

    trainer = _get_trainer(config)

    try:
        trainer.load_bot(bot_id)
    except Exception as e:
        click.secho(f"❌ Failed to load bot {bot_id}: {e}", fg="red")
        raise SystemExit(1)

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            click.echo(f"⏳ Ingesting {path}...")

            if path.startswith("http://") or path.startswith("https://"):
                trainer.add_document_from_link([path], bot_id)
            else:
                import os
                if not os.path.exists(path):
                    click.secho(f"❌ File {path!r} does not exist.", fg="red")
                    raise SystemExit(1)
                trainer.add_document_from_path(path, bot_id)

            trainer._rebuild_bot(bot_id)
            click.secho(f"✅ Document added: {path}", fg="green")
            click.echo(f"   Bot: {bot_id}")
            click.echo(f"\nChat with your bot: longtrainer chat {bot_id}\n")
            break

        except LongTrainerRateLimitError as e:
            if attempt >= max_retries - 1:
                click.secho("❌ Rate limit exceeded. Please try again later.", fg="red")
                raise SystemExit(1)
            wait_time = int(e.retry_after) + 1
            click.secho(
                f"⏳ Rate limit hit ({e.resource}). Retrying in {wait_time}s... "
                f"(attempt {attempt + 1}/{max_retries})",
                fg="yellow",
            )
            with click.progressbar(range(wait_time), label="Waiting") as bar:
                for _ in bar:
                    _time.sleep(1)


# ─── chat ─────────────────────────────────────────────────────────────────────

@cli.command("chat")
@click.argument("bot_id")
@click.option("--config", "-c", default="longtrainer.yaml", help="Path to config file.")
def chat_command(bot_id: str, config: str) -> None:
    """Start an interactive terminal chat with a bot."""
    import time
    from longtrainer.rate_limiter import LongTrainerRateLimitError

    trainer = _get_trainer(config)

    click.echo(f"⏳ Loading bot {bot_id}...")
    try:
        trainer.load_bot(bot_id)
    except Exception as e:
        click.secho(f"❌ Failed to load bot {bot_id}: {e}", fg="red")
        raise SystemExit(1)

    chat_id = trainer.new_chat(bot_id)

    click.secho("\n💬 LongTrainer Chat", fg="cyan", bold=True)
    click.echo(f"   Bot    : {bot_id}")
    click.echo(f"   Chat ID: {chat_id}")
    click.echo("\nType 'exit' or 'quit' to end the session.\n")

    while True:
        try:
            query = click.prompt("You", prompt_suffix="> ")
            if not query or query.lower() in ("exit", "quit"):
                break

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    result = trainer.get_response(query, bot_id, chat_id)
                    response = result[0] if isinstance(result, tuple) else str(result)
                    click.secho("\nLongTrainer> ", fg="blue", bold=True, nl=False)
                    click.echo(response)
                    click.echo()
                    break

                except LongTrainerRateLimitError as e:
                    if attempt >= max_retries - 1:
                        click.secho("❌ Rate limit exceeded. Please try again later.", fg="red")
                        raise SystemExit(1)
                    wait_time = int(e.retry_after) + 1
                    click.secho(
                        f"⏳ Rate limit hit ({e.resource}). Retrying in {wait_time}s... "
                        f"(attempt {attempt + 1}/{max_retries})",
                        fg="yellow",
                    )
                    with click.progressbar(range(wait_time), label="Waiting") as bar:
                        for _ in bar:
                            time.sleep(1)

        except (KeyboardInterrupt, EOFError):
            click.echo()
            break

    click.secho("\n👋 Chat session ended.\n", fg="yellow")


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
