"""Test 1: Import & Package Verification

Verifies that longtrainer is installed correctly and all modules import.
No external services required.
"""

import sys

def test_imports():
    """Test that all V2 modules are importable."""
    print("=" * 60)
    print("TEST 1: Import & Package Verification")
    print("=" * 60)

    checks = []

    # Core package
    try:
        import longtrainer
        checks.append(("longtrainer", True, f"v{longtrainer.__version__}"))
    except ImportError as e:
        checks.append(("longtrainer", False, str(e)))

    # Trainer
    try:
        from longtrainer.trainer import LongTrainer
        checks.append(("LongTrainer", True, ""))
    except ImportError as e:
        checks.append(("LongTrainer", False, str(e)))

    # Bot modules
    try:
        from longtrainer.bot import RAGBot, AgentBot
        checks.append(("RAGBot + AgentBot", True, ""))
    except ImportError as e:
        checks.append(("RAGBot + AgentBot", False, str(e)))

    # Tools
    try:
        from longtrainer.tools import ToolRegistry, web_search, document_reader, get_builtin_tools
        tools = get_builtin_tools()
        checks.append(("ToolRegistry + built-in tools", True, f"{[t.name for t in tools]}"))
    except ImportError as e:
        checks.append(("ToolRegistry + built-in tools", False, str(e)))

    # Loaders
    try:
        from longtrainer.loaders import DocumentLoader, TextSplitter
        checks.append(("DocumentLoader + TextSplitter", True, ""))
    except ImportError as e:
        checks.append(("DocumentLoader + TextSplitter", False, str(e)))

    # Retrieval
    try:
        from longtrainer.retrieval import DocumentRetriever, MultiQueryEnsembleRetriever
        checks.append(("DocumentRetriever + MultiQueryEnsembleRetriever", True, ""))
    except ImportError as e:
        checks.append(("DocumentRetriever + MultiQueryEnsembleRetriever", False, str(e)))

    # Vision
    try:
        from longtrainer.vision_bot import VisionBot, VisionMemory
        checks.append(("VisionBot + VisionMemory", True, ""))
    except ImportError as e:
        checks.append(("VisionBot + VisionMemory", False, str(e)))

    # Utils
    try:
        from longtrainer.utils import serialize_document, deserialize_document, LineListOutputParser
        checks.append(("Utils (serialize/deserialize/parser)", True, ""))
    except ImportError as e:
        checks.append(("Utils", False, str(e)))

    # LangGraph (optional agent dependency)
    try:
        from langgraph.prebuilt import create_react_agent
        checks.append(("langgraph (agent mode)", True, ""))
    except ImportError:
        checks.append(("langgraph (agent mode)", False, "Not installed — agent mode unavailable"))

    # Print results
    print()
    passed = 0
    failed = 0
    for name, ok, detail in checks:
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
    success = test_imports()
    sys.exit(0 if success else 1)
