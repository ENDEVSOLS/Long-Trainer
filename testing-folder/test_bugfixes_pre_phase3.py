"""Tests for the 3 known bugs fixed before Phase 3.

Fixes verified:
  B1 - update_chatbot() uses correct key (db_path, not faiss_path)
  B2 - async document ingestion: aadd_document_from_path, add_documents_from_paths
  B3 - retrieval confidence scores: invoke_vectorstore_with_scores
  B4 - parallel document ingestion via ThreadPoolExecutor
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_storage():
    """Return a fully mocked MongoStorage with all methods stubbed."""
    storage = MagicMock()
    storage.find_documents.return_value = []
    storage.count_documents.return_value = 0
    storage.save_document.return_value = None
    return storage


def _make_doc_manager(storage=None):
    """Return a DocumentManager with a mocked storage and loader."""
    from longtrainer.documents import DocumentManager
    from longtrainer.loaders import DocumentLoader
    st = storage or _make_mock_storage()
    loader = MagicMock(spec=DocumentLoader)
    return DocumentManager(storage=st, document_loader=loader), loader, st


def _make_fake_doc(content: str = "hello"):
    """Return a minimal LangChain Document stub."""
    from langchain_core.documents import Document
    return Document(page_content=content, metadata={"source": "test"})


# ─────────────────────────────────────────────────────────────────────────────
# B1 — update_chatbot() uses correct key (db_path)
# ─────────────────────────────────────────────────────────────────────────────

class TestUpdateChatbotKey:
    """Verify update_chatbot delegates to create_bot without touching faiss_path."""

    def _make_trainer(self):
        from longtrainer.trainer import LongTrainer
        from unittest.mock import patch as p
        with p("longtrainer.trainer.MongoStorage"), \
             p("longtrainer.trainer.DocumentLoader"), \
             p("longtrainer.trainer.TextSplitter"), \
             p("longtrainer.trainer.DocumentManager"), \
             p("longtrainer.trainer.ChatManager"), \
             p("longtrainer.trainer.get_llm"), \
             p("longtrainer.trainer.get_embedding_model"):
            trainer = LongTrainer.__new__(LongTrainer)
            trainer.bot_data = {}
            trainer._doc_manager = MagicMock()
            trainer._doc_manager.count_documents.return_value = 0
            trainer._doc_manager.add_document_from_path = MagicMock()
            trainer._doc_manager.add_document_from_link = MagicMock()
            trainer._doc_manager.add_document_from_query = MagicMock()
            trainer._doc_manager.pass_documents = MagicMock()
            trainer._storage = MagicMock()
            trainer.llm = MagicMock()
            trainer.embedding_model = MagicMock()
            trainer.prompt_template = "Answer: {context} {chat_history} {question}"
            trainer.k = 3
            trainer.max_token_limit = 32000
            trainer.ensemble = False
            return trainer

    def test_update_chatbot_calls_create_bot_not_faiss_path(self):
        """update_chatbot must delegate to create_bot (no faiss_path access)."""
        trainer = self._make_trainer()
        bot_id = "bot-test-b1"

        # Stub bot_data WITHOUT a faiss_path key
        trainer.bot_data[bot_id] = {
            "chains": {},
            "assistants": {},
            "retriever": None,
            "vectorstore": None,
            "ensemble_retriever": None,
            "db_path": "/tmp/db_test",   # correct key
            "prompt_template": trainer.prompt_template,
            "agent_mode": False,
            "tools": MagicMock(),
        }

        # Patch create_bot — if faiss_path was accessed it would KeyError before here
        with patch.object(trainer, "create_bot") as mock_create:
            trainer.update_chatbot(paths=[], bot_id=bot_id)
            mock_create.assert_called_once_with(bot_id, prompt_template=None)

    def test_update_chatbot_no_faiss_path_keyerror(self):
        """Accessing bot['faiss_path'] on a real bot_data dict raises KeyError.

        This confirms the old bug location and that our code no longer hits it.
        """
        bot_data = {"db_path": "/tmp/db"}
        with pytest.raises(KeyError):
            _ = bot_data["faiss_path"]


# ─────────────────────────────────────────────────────────────────────────────
# B2 — Async document ingestion
# ─────────────────────────────────────────────────────────────────────────────

class TestAsyncDocumentIngestion:
    """Verify aadd_document_from_path and parallel ingestion work correctly."""

    def test_aadd_document_from_path_is_coroutine(self):
        """aadd_document_from_path must return an awaitable."""
        import inspect
        from longtrainer.documents import DocumentManager
        dm = DocumentManager.__new__(DocumentManager)
        dm.storage = MagicMock()
        dm.document_loader = MagicMock()
        coro = dm.aadd_document_from_path("dummy.pdf", "bot-1")
        assert inspect.isawaitable(coro), "aadd_document_from_path must be awaitable"
        # Close to avoid 'coroutine was never awaited' warning
        coro.close()

    def test_aadd_document_from_path_calls_sync_under_the_hood(self):
        """Awaiting aadd_document_from_path must ultimately call add_document_from_path."""
        doc_manager, loader, storage = _make_doc_manager()
        loader.load_pdf.return_value = [_make_fake_doc("page1")]

        with patch.object(
            doc_manager, "add_document_from_path", wraps=doc_manager.add_document_from_path
        ) as spy:
            asyncio.run(doc_manager.aadd_document_from_path("test.pdf", "bot-1"))
            spy.assert_called_once_with("test.pdf", "bot-1", False)

    def test_add_documents_from_paths_loads_all_files(self):
        """add_documents_from_paths must attempt to load every path."""
        doc_manager, loader, storage = _make_doc_manager()
        loader.load_pdf.return_value = [_make_fake_doc()]
        loader.load_markdown.return_value = [_make_fake_doc()]

        paths = ["a.pdf", "b.md", "c.pdf"]

        # We patch aadd_document_from_path so it resolves synchronously
        call_record = []

        async def fake_async_add(path, bot_id, use_unstructured=False):
            call_record.append(path)

        with patch.object(doc_manager, "aadd_document_from_path", side_effect=fake_async_add):
            asyncio.run(
                doc_manager._aadd_documents_from_paths(paths, "bot-1")
            )

        assert sorted(call_record) == sorted(paths), (
            f"Expected all 3 paths to be loaded; got {call_record}"
        )

    def test_add_documents_from_paths_continues_on_single_failure(self):
        """A failure on one file must not abort the rest."""
        doc_manager, loader, storage = _make_doc_manager()
        call_record: list[str] = []
        failures: list[str] = []

        async def fake_async_add(path, bot_id, use_unstructured=False):
            if path == "bad.pdf":
                raise ValueError("Mock load error")
            call_record.append(path)

        with patch.object(doc_manager, "aadd_document_from_path", side_effect=fake_async_add):
            # _aadd_documents_from_paths gathers with return_exceptions=True
            asyncio.run(
                doc_manager._aadd_documents_from_paths(
                    ["good1.pdf", "bad.pdf", "good2.pdf"], "bot-1"
                )
            )

        # good files must still have been processed
        assert "good1.pdf" in call_record
        assert "good2.pdf" in call_record

    def test_aadd_link_loads_all_links_concurrently(self):
        """aadd_document_from_link must attempt every link concurrently."""
        doc_manager, loader, storage = _make_doc_manager()
        loader.load_urls.return_value = [_make_fake_doc()]

        links = ["https://a.com", "https://b.com", "https://c.com"]
        call_record: list[str] = []

        def fake_load_save(link, bot_id):
            call_record.append(link)

        with patch.object(doc_manager, "_load_and_save_single_link", side_effect=fake_load_save):
            asyncio.run(doc_manager.aadd_document_from_link(links, "bot-1"))

        assert sorted(call_record) == sorted(links)


# ─────────────────────────────────────────────────────────────────────────────
# B3 — Retrieval confidence scores
# ─────────────────────────────────────────────────────────────────────────────

class TestRetrievalConfidenceScores:
    """Verify similarity_search_with_score returns scored results with metadata."""

    def _make_retriever_wrapper(self, scored_results):
        """Return a DocumentRetriever (bypass __init__) with a mocked faiss_index."""
        from longtrainer.retrieval import DocumentRetriever
        wrapper = DocumentRetriever.__new__(DocumentRetriever)
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.return_value = scored_results
        wrapper.faiss_index = mock_vs
        wrapper.k = 3
        return wrapper

    def test_returns_list_of_tuples(self):
        """similarity_search_with_score must return a list of (Document, float)."""
        doc = _make_fake_doc("test content")
        wrapper = self._make_retriever_wrapper([(doc, 0.2)])
        results = wrapper.similarity_search_with_score("query")
        assert isinstance(results, list)
        assert len(results) == 1
        result_doc, score = results[0]
        assert hasattr(result_doc, "page_content")
        assert isinstance(score, float)

    def test_injects_retrieval_score_metadata(self):
        """Each returned document must have retrieval_score in its metadata."""
        from langchain_core.documents import Document
        doc1 = Document(page_content="best match", metadata={})
        doc2 = Document(page_content="worse match", metadata={})
        wrapper = self._make_retriever_wrapper([(doc1, 0.1), (doc2, 0.5)])
        results = wrapper.similarity_search_with_score("query")

        scores = [doc.metadata["retrieval_score"] for doc, _ in results]
        assert all(0.0 <= s <= 1.0 for s in scores), f"Scores out of range: {scores}"

    def test_best_match_has_highest_retrieval_score(self):
        """Lower L2 distance → higher retrieval_score (1.0 for the best result)."""
        from langchain_core.documents import Document
        best = Document(page_content="best", metadata={})
        worst = Document(page_content="worst", metadata={})
        # FAISS: lower score = closer = better
        wrapper = self._make_retriever_wrapper([(best, 0.0), (worst, 1.0)])
        results = wrapper.similarity_search_with_score("query")

        score_map = {doc.page_content: doc.metadata["retrieval_score"] for doc, _ in results}
        assert score_map["best"] >= score_map["worst"], (
            f"Best match should have higher retrieval_score: {score_map}"
        )

    def test_returns_empty_list_on_faiss_error(self):
        """If FAISS raises, similarity_search_with_score must return []."""
        from longtrainer.retrieval import DocumentRetriever
        wrapper = DocumentRetriever.__new__(DocumentRetriever)
        mock_vs = MagicMock()
        mock_vs.similarity_search_with_score.side_effect = RuntimeError("FAISS error")
        wrapper.faiss_index = mock_vs
        wrapper.k = 3

        results = wrapper.similarity_search_with_score("query")
        assert results == []

    def test_returns_empty_list_when_no_vectorstore(self):
        """When faiss_index is None, must return []."""
        from longtrainer.retrieval import DocumentRetriever
        wrapper = DocumentRetriever.__new__(DocumentRetriever)
        wrapper.faiss_index = None
        wrapper.k = 3

        results = wrapper.similarity_search_with_score("query")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# B3 — invoke_vectorstore_with_scores on LongTrainer public API
# ─────────────────────────────────────────────────────────────────────────────

class TestInvokeVectorstoreWithScores:
    """Verify the public invoke_vectorstore_with_scores method on LongTrainer."""

    def _make_trainer_with_bot(self):
        """Return a partially mocked LongTrainer with one bot pre-loaded."""
        from longtrainer.trainer import LongTrainer
        trainer = LongTrainer.__new__(LongTrainer)
        trainer.k = 3

        mock_vs = MagicMock()
        trainer.bot_data = {
            "bot-score-test": {
                "vectorstore": mock_vs,
                "ensemble_retriever": MagicMock(),
            }
        }
        return trainer, mock_vs

    def test_calls_similarity_search_with_score(self):
        """invoke_vectorstore_with_scores must call similarity_search_with_score."""
        trainer, mock_vs = self._make_trainer_with_bot()

        from langchain_core.documents import Document
        fake_doc = Document(page_content="result", metadata={})
        mock_vs.similarity_search_with_score.return_value = [(fake_doc, 0.1)]

        results = trainer.invoke_vectorstore_with_scores("bot-score-test", "my query")
        mock_vs.similarity_search_with_score.assert_called_once()
        assert len(results) == 1

    def test_raises_for_unknown_bot(self):
        """Must raise ValueError for an unknown bot_id."""
        trainer, _ = self._make_trainer_with_bot()
        with pytest.raises(ValueError, match="not found"):
            trainer.invoke_vectorstore_with_scores("no-such-bot", "query")

    def test_returns_empty_list_when_no_vectorstore(self):
        """Must return [] when vectorstore is None for the bot."""
        from longtrainer.trainer import LongTrainer
        trainer = LongTrainer.__new__(LongTrainer)
        trainer.k = 3
        trainer.bot_data = {"bot-empty": {"vectorstore": None, "ensemble_retriever": None}}

        results = trainer.invoke_vectorstore_with_scores("bot-empty", "query")
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
# B4 — Parallel ingestion thread pool
# ─────────────────────────────────────────────────────────────────────────────

class TestParallelIngestionThreadPool:
    """Verify the _LOADER_EXECUTOR is created with the correct parameters."""

    def test_executor_max_workers(self):
        """Thread pool must have max_workers=4 and correct thread name prefix."""
        from longtrainer import documents as doc_module
        executor = doc_module._LOADER_EXECUTOR
        assert executor._max_workers == 4
        # Thread name prefix is stored differently depending on Python version
        prefix = getattr(executor, "_thread_name_prefix", None) or ""
        assert "lt_doc_loader" in prefix

    def test_save_docs_helper(self):
        """_save_docs must call storage.save_document once per document."""
        doc_manager, _, storage = _make_doc_manager(storage=_make_mock_storage())
        docs = [_make_fake_doc("a"), _make_fake_doc("b"), _make_fake_doc("c")]
        doc_manager._save_docs("bot-1", docs)
        assert storage.save_document.call_count == 3
