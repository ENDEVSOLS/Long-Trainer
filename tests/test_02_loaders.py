"""Test 2: Document Loading & Text Splitting

Tests the document loading pipeline WITHOUT any LLM or MongoDB.
Creates sample files, loads them, splits them, and verifies output.
"""

import os
import sys
import tempfile


def test_document_loading():
    """Test document loading for various file types."""
    print("=" * 60)
    print("TEST 2: Document Loading & Text Splitting")
    print("=" * 60)

    from longtrainer.loaders import DocumentLoader, TextSplitter

    loader = DocumentLoader()
    splitter = TextSplitter(chunk_size=200, chunk_overlap=50)
    results = []

    # Create temp files for testing
    with tempfile.TemporaryDirectory() as tmpdir:

        # 1. TXT/Markdown file
        txt_path = os.path.join(tmpdir, "sample.txt")
        with open(txt_path, "w") as f:
            f.write("LongTrainer is a production-ready RAG framework.\n" * 20)
            f.write("It supports multi-bot management and streaming responses.\n" * 20)

        try:
            docs = loader.load_markdown(txt_path)
            assert len(docs) > 0, "No documents loaded from TXT"
            assert docs[0].page_content, "Empty page_content"
            results.append(("TXT/Markdown loading", True, f"{len(docs)} docs"))
        except Exception as e:
            results.append(("TXT/Markdown loading", False, str(e)))

        # 2. CSV file
        csv_path = os.path.join(tmpdir, "sample.csv")
        with open(csv_path, "w") as f:
            f.write("question,answer\n")
            f.write("What is LongTrainer?,A production-ready RAG framework\n")
            f.write("What does it support?,Multi-bot management and streaming\n")
            f.write("What is the default LLM?,ChatOpenAI gpt-4o\n")

        try:
            docs = loader.load_csv(csv_path)
            assert len(docs) > 0, "No documents loaded from CSV"
            results.append(("CSV loading", True, f"{len(docs)} docs"))
        except Exception as e:
            results.append(("CSV loading", False, str(e)))

        # 3. HTML file
        html_path = os.path.join(tmpdir, "sample.html")
        with open(html_path, "w") as f:
            f.write("<html><body><h1>LongTrainer</h1><p>A RAG framework for production use.</p></body></html>")

        try:
            docs = loader.load_text_from_html(html_path)
            assert len(docs) > 0, "No documents loaded from HTML"
            results.append(("HTML loading", True, f"{len(docs)} docs"))
        except Exception as e:
            results.append(("HTML loading", False, str(e)))

        # 4. Text splitting
        try:
            txt_docs = loader.load_markdown(txt_path)
            splits = splitter.split_documents(txt_docs)
            assert len(splits) > len(txt_docs), "Splitting should create more chunks"
            for s in splits:
                assert len(s.page_content) <= 200 + 50, f"Chunk too large: {len(s.page_content)}"
            results.append(("Text splitting", True, f"{len(txt_docs)} docs → {len(splits)} chunks"))
        except Exception as e:
            results.append(("Text splitting", False, str(e)))

        # 5. Serialization round-trip
        try:
            from longtrainer.utils import serialize_document, deserialize_document
            from langchain_core.documents import Document

            original = Document(page_content="Test content", metadata={"source": "test.txt"})
            serialized = serialize_document(original)
            restored = deserialize_document(serialized)
            assert restored.page_content == original.page_content
            assert restored.metadata == original.metadata
            results.append(("Serialize/deserialize round-trip", True, ""))
        except Exception as e:
            results.append(("Serialize/deserialize round-trip", False, str(e)))

    # Print results
    print()
    passed = 0
    failed = 0
    for name, ok, detail in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        detail_str = f" — {detail}" if detail else ""
        print(f"  {status}: {name}{detail_str}")
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = test_document_loading()
    sys.exit(0 if success else 1)
