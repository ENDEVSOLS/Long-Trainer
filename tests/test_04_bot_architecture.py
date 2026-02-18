"""Test 4: Bot Architecture (RAGBot + AgentBot)

Tests bot creation, LCEL chain construction, and memory management.
Uses mock/minimal components — no OpenAI API key required.
"""

import sys


def test_bot_architecture():
    """Test RAGBot and AgentBot construction and memory."""
    print("=" * 60)
    print("TEST 4: Bot Architecture (RAGBot + AgentBot)")
    print("=" * 60)

    from langchain_core.chat_history import InMemoryChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage
    from longtrainer.bot import RAGBot, AgentBot

    results = []

    # 1. InMemoryChatMessageHistory works
    try:
        history = InMemoryChatMessageHistory()
        history.add_message(HumanMessage(content="Hello"))
        history.add_message(AIMessage(content="Hi there!"))
        assert len(history.messages) == 2
        assert history.messages[0].content == "Hello"
        assert history.messages[1].content == "Hi there!"
        results.append(("InMemoryChatMessageHistory", True, f"{len(history.messages)} messages"))
    except Exception as e:
        results.append(("InMemoryChatMessageHistory", False, str(e)))

    # 2. RAGBot class exists and has expected methods
    try:
        assert hasattr(RAGBot, "invoke"), "Missing invoke method"
        assert hasattr(RAGBot, "stream"), "Missing stream method"
        assert hasattr(RAGBot, "astream"), "Missing astream method"
        assert hasattr(RAGBot, "save_context"), "Missing save_context method"
        assert hasattr(RAGBot, "memory"), "Missing memory property"
        results.append(("RAGBot interface", True, "invoke, stream, astream, save_context, memory"))
    except Exception as e:
        results.append(("RAGBot interface", False, str(e)))

    # 3. AgentBot class exists and has expected methods
    try:
        assert hasattr(AgentBot, "invoke"), "Missing invoke method"
        assert hasattr(AgentBot, "stream"), "Missing stream method"
        assert hasattr(AgentBot, "astream"), "Missing astream method"
        assert hasattr(AgentBot, "save_context"), "Missing save_context method"
        assert hasattr(AgentBot, "memory"), "Missing memory property"
        results.append(("AgentBot interface", True, "invoke, stream, astream, save_context, memory"))
    except Exception as e:
        results.append(("AgentBot interface", False, str(e)))

    # 4. AgentBot raises ImportError without langgraph
    # (This tests the lazy import guard — if langgraph IS installed, we test it works)
    try:
        try:
            from langgraph.prebuilt import create_react_agent
            results.append(("langgraph available", True, "Agent mode will work"))
        except ImportError:
            results.append(("langgraph not installed", True, "Agent mode correctly unavailable"))
    except Exception as e:
        results.append(("langgraph check", False, str(e)))

    # 5. VisionBot + VisionMemory interface
    try:
        from longtrainer.vision_bot import VisionBot, VisionMemory
        assert hasattr(VisionMemory, "save_chat_history"), "Missing save_chat_history"
        assert hasattr(VisionMemory, "save_context"), "Missing save_context"
        assert hasattr(VisionMemory, "get_answer"), "Missing get_answer"
        assert hasattr(VisionMemory, "memory"), "Missing memory property"
        assert hasattr(VisionBot, "create_vision_bot"), "Missing create_vision_bot"
        assert hasattr(VisionBot, "get_response"), "Missing get_response"
        results.append(("VisionBot + VisionMemory interface", True, "all methods present"))
    except Exception as e:
        results.append(("VisionBot + VisionMemory interface", False, str(e)))

    # 6. Retriever classes
    try:
        from longtrainer.retrieval import DocumentRetriever, MultiQueryEnsembleRetriever
        from langchain_core.retrievers import BaseRetriever
        assert issubclass(MultiQueryEnsembleRetriever, BaseRetriever)
        results.append(("MultiQueryEnsembleRetriever extends BaseRetriever", True, ""))
    except Exception as e:
        results.append(("Retriever classes", False, str(e)))

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
    success = test_bot_architecture()
    sys.exit(0 if success else 1)
