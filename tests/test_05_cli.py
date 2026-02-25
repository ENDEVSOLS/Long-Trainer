import os
from unittest.mock import patch

from click.testing import CliRunner

from longtrainer.cli import cli


def test_cli_version():
    """Test the CLI version command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "longtrainer, version" in result.output


def test_cli_help():
    """Test the CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "init" in result.output
    assert "serve" in result.output


def test_cli_init_command(tmp_path):
    """Test non-interactive init command config generation."""
    test_yaml = tmp_path / "longtrainer.yaml"
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "init",
            "--mongo", "mongodb://test:27017/",
            "--llm-provider", "openai",
            "--model-name", "gpt-4o-mini",
            "--embedding-provider", "openai",
            "--embedding-model", "text-embedding-3-small",
            "--vectorstore-provider", "pinecone",
            "--chunk-size", "1024",
            "--chunk-overlap", "100",
            "--encrypt-chats",
            "-o", str(test_yaml),
        ]
    )

    assert result.exit_code == 0
    assert "Config written to" in result.output
    assert test_yaml.exists()

    # Verify generated YAML content
    import yaml
    with open(test_yaml) as f:
        config = yaml.safe_load(f)

    assert config["mongo_endpoint"] == "mongodb://test:27017/"
    assert config["llm"]["provider"] == "openai"
    assert config["llm"]["model_name"] == "gpt-4o-mini"
    assert config["embedding"]["provider"] == "openai"
    assert config["embedding"]["model_name"] == "text-embedding-3-small"
    assert config["vector_store"]["provider"] == "pinecone"
    assert config["chunking"]["chunk_size"] == 1024
    assert config["chunking"]["chunk_overlap"] == 100
    assert config["encrypt_chats"] is True
