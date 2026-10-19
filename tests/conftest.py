"""Explicit selection of tests requiring local integration services."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests requiring MongoDB and the integration extra.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(
        reason="Requires MongoDB and .[integration]; CI runs this with --run-integration.",
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)
