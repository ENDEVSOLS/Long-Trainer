"""Exercise optional tool wrappers against their installed client APIs."""
from datetime import datetime
from types import SimpleNamespace

import pytest


def test_arxiv_wrapper_uses_compatible_client_api(monkeypatch):
    arxiv = pytest.importorskip("arxiv")
    from longtrainer.agent_types import resolve_tools

    paper = SimpleNamespace(
        updated=datetime(2017, 6, 12),
        title="Attention Is All You Need",
        authors=[SimpleNamespace(name="Test Author")],
        summary="Transformer architecture regression fixture.",
        entry_id="https://arxiv.org/abs/1706.03762",
    )
    searches = []

    def results(self, search, **kwargs):
        searches.append(search)
        return iter([paper])

    monkeypatch.setattr(arxiv.Client, "results", results)
    tool = resolve_tools(["arxiv"])[0]
    response = tool.invoke("1706.03762")
    assert "Attention Is All You Need" in response
    assert "Transformer architecture" in response
    assert searches[0].id_list == ["1706.03762"]
