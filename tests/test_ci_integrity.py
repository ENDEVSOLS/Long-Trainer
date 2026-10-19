"""Regressions for failures previously swallowed by aggregate script tests."""

import pytest
import test_02_loaders as loader_checks
import test_10_lazy_loading as integration_checks

from longtrainer.loaders import DocumentLoader


def test_failed_loader_subcheck_fails_pytest(monkeypatch):
    monkeypatch.setattr(DocumentLoader, "load_csv", lambda self, path: [])

    with pytest.raises(AssertionError, match="document loading checks failed"):
        loader_checks.test_document_loading()


def test_failed_integration_setup_fails_pytest(monkeypatch, tmp_path):
    monkeypatch.setattr(integration_checks, "_run_lazy_loading_checks", lambda: False)

    with pytest.raises(AssertionError, match="Lazy-loading integration checks failed"):
        integration_checks.test_lazy_loading(monkeypatch, tmp_path)
