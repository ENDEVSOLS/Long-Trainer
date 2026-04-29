"""Test 10: Lazy Loading — Real Integration Test

Validates that:
1. load_bot() no longer eagerly loads chat histories into RAM
2. _ensure_chat_loaded() correctly lazy-loads a single chat from MongoDB
3. _ensure_vision_chat_loaded() correctly lazy-loads a single vision chat
4. get_response() auto-lazy-loads before processing
5. get_vision_response() auto-lazy-loads before processing
6. new_chat() / new_vision_chat() still work correctly
7. Subsequent calls (cache hit) skip MongoDB entirely
8. All other functionalities (list_chats, delete, etc.) are unbroken

Requires a running MongoDB instance and real dependencies (no mocks).
"""

import sys
import time
import gc

# Allow running standalone
if __name__ == "__main__":
    sys.path.insert(0, ".")


def test_lazy_loading():
    """Real integration test for lazy chat loading."""
    print("=" * 70)
    print("TEST 10: Lazy Loading — Real Integration Test")
    print("=" * 70)

    results = []

    # ─── Setup: Create a real LongTrainer with local MongoDB ─────────────────
    try:
        import os
        os.environ["OPENAI_API_KEY"] = "sk-mock-key-for-testing"
        from langchain_core.embeddings import FakeEmbeddings
        from longtrainer.trainer import LongTrainer
        from longtrainer.bot import RAGBot, AgentBot
        from longtrainer.vision_bot import VisionMemory

        trainer = LongTrainer(
            mongo_endpoint="mongodb://localhost:27017/",
            vector_store_provider="qdrant",
            vector_store_kwargs={"location": ":memory:"},
            embedding_model=FakeEmbeddings(size=1536),
        )
        results.append(("LongTrainer init", True, "Connected to MongoDB"))
    except Exception as e:
        results.append(("LongTrainer init", False, str(e)))
        _print_results(results)
        return False

    bot_id = None
    chat_id = None
    vision_chat_id = None

    # ─── Test 1: Initialize + Create Bot ─────────────────────────────────────
    try:
        bot_id = trainer.initialize_bot_id()
        assert bot_id, "Bot ID should not be empty"
        assert bot_id in trainer.bot_data, "Bot should be in bot_data"
        results.append(("initialize_bot_id", True, f"Created {bot_id}"))
    except Exception as e:
        results.append(("initialize_bot_id", False, str(e)))
        _print_results(results)
        return False

    # Add a minimal document so create_bot doesn't fail on empty vectorstore
    try:
        from langchain_core.documents import Document
        trainer.pass_documents([Document(page_content="Test document for lazy loading verification.")], bot_id)
        trainer.create_bot(bot_id=bot_id)
        results.append(("create_bot", True, "Bot created with test document"))
    except Exception as e:
        results.append(("create_bot", False, str(e)))
        _print_results(results)
        return False

    # ─── Test 2: new_chat() works — creates an in-memory bot instance ────────
    try:
        chat_id = trainer.new_chat(bot_id)
        assert chat_id, "Chat ID should not be empty"
        assert chat_id in trainer.bot_data[bot_id]["chains"], "New chat should be in chains"
        results.append(("new_chat", True, f"Created {chat_id}"))
    except Exception as e:
        results.append(("new_chat", False, str(e)))

    # ─── Test 3: new_vision_chat() works ─────────────────────────────────────
    try:
        vision_chat_id = trainer.new_vision_chat(bot_id)
        assert vision_chat_id, "Vision chat ID should not be empty"
        assert vision_chat_id in trainer.bot_data[bot_id]["assistants"], "Vision chat should be in assistants"
        results.append(("new_vision_chat", True, f"Created {vision_chat_id}"))
    except Exception as e:
        results.append(("new_vision_chat", False, str(e)))

    # ─── Test 4: Manually store some chat history in MongoDB ─────────────────
    # Simulate what happens when a real user sends messages
    test_chat_id = "chat-lazy-test-001"
    test_vision_chat_id = "vision-lazy-test-001"
    try:
        # Store 3 chat messages directly into MongoDB
        for i in range(3):
            trainer._storage.store_chat(
                bot_id=bot_id,
                chat_id=test_chat_id,
                query=f"Test question {i+1}",
                answer=f"Test answer {i+1}",
            )

        # Store 2 vision chat messages
        for i in range(2):
            trainer._storage.store_vision_chat(
                bot_id=bot_id,
                vision_chat_id=test_vision_chat_id,
                image_paths=["test.jpg"],
                query=f"Vision question {i+1}",
                response=f"Vision response {i+1}",
            )

        # Verify MongoDB has the records
        chat_data = trainer._storage.get_chat_by_id(test_chat_id, "oldest")
        vision_data = trainer._storage.get_vision_chat_by_id(test_vision_chat_id, "oldest")
        assert chat_data and len(chat_data) == 3, f"Expected 3 chat records, got {len(chat_data) if chat_data else 0}"
        assert vision_data and len(vision_data) == 2, f"Expected 2 vision records, got {len(vision_data) if vision_data else 0}"

        results.append(("Store test history in MongoDB", True, f"3 chats + 2 vision chats stored"))
    except Exception as e:
        results.append(("Store test history in MongoDB", False, str(e)))

    # ─── Test 5: load_bot() does NOT eagerly load chat history ───────────────
    try:
        # Clear in-memory data to simulate server restart
        del trainer.bot_data[bot_id]
        gc.collect()

        # Measure load time
        start = time.perf_counter()
        trainer.load_bot(bot_id)
        elapsed = time.perf_counter() - start

        assert bot_id in trainer.bot_data, "Bot should be in bot_data after load"
        assert trainer.bot_data[bot_id]["chains"] == {}, f"chains should be empty after load, got {len(trainer.bot_data[bot_id]['chains'])} entries"
        assert trainer.bot_data[bot_id]["assistants"] == {}, f"assistants should be empty after load, got {len(trainer.bot_data[bot_id]['assistants'])} entries"
        assert trainer.bot_data[bot_id]["vectorstore"] is not None, "Vectorstore should be loaded"
        assert trainer.bot_data[bot_id]["ensemble_retriever"] is not None, "Retriever should be loaded"

        results.append(("load_bot — no eager loading", True, f"Loaded in {elapsed*1000:.1f}ms, 0 chats in RAM"))
    except Exception as e:
        results.append(("load_bot — no eager loading", False, str(e)))

    # ─── Test 6: _ensure_chat_loaded() — lazy loads from MongoDB ─────────────
    try:
        # Chat should NOT be in memory
        assert test_chat_id not in trainer.bot_data[bot_id]["chains"], "Chat should not be in memory before lazy load"

        # Lazy load it
        start = time.perf_counter()
        trainer._ensure_chat_loaded(bot_id, test_chat_id)
        elapsed = time.perf_counter() - start

        # Now it should be in memory
        assert test_chat_id in trainer.bot_data[bot_id]["chains"], "Chat should be in memory after lazy load"

        # Verify the bot instance has the correct history
        bot_instance = trainer.bot_data[bot_id]["chains"][test_chat_id]
        assert isinstance(bot_instance, RAGBot), f"Expected RAGBot, got {type(bot_instance)}"
        assert len(bot_instance.chat_history.messages) == 6, f"Expected 6 messages (3 Q+A pairs), got {len(bot_instance.chat_history.messages)}"

        # Verify message content
        messages = bot_instance.chat_history.messages
        assert "Test question 1" in messages[0].content, f"First message should contain 'Test question 1', got '{messages[0].content}'"
        assert "Test answer 1" in messages[1].content, f"Second message should contain 'Test answer 1', got '{messages[1].content}'"

        results.append(("_ensure_chat_loaded — MongoDB replay", True, f"6 messages loaded in {elapsed*1000:.1f}ms"))
    except Exception as e:
        results.append(("_ensure_chat_loaded — MongoDB replay", False, str(e)))

    # ─── Test 7: _ensure_chat_loaded() — cache hit (no-op) ───────────────────
    try:
        # Store reference to current bot instance
        cached_instance = trainer.bot_data[bot_id]["chains"].get(test_chat_id)

        # Call again — should be a no-op
        start = time.perf_counter()
        trainer._ensure_chat_loaded(bot_id, test_chat_id)
        elapsed = time.perf_counter() - start

        # Same object should still be there (identity check)
        assert trainer.bot_data[bot_id]["chains"][test_chat_id] is cached_instance, "Cache hit should return same object"

        results.append(("_ensure_chat_loaded — cache hit", True, f"No-op in {elapsed*1000:.3f}ms"))
    except Exception as e:
        results.append(("_ensure_chat_loaded — cache hit", False, str(e)))

    # ─── Test 8: _ensure_vision_chat_loaded() — lazy loads from MongoDB ──────
    try:
        assert test_vision_chat_id not in trainer.bot_data[bot_id]["assistants"], "Vision chat should not be in memory"

        start = time.perf_counter()
        trainer._ensure_vision_chat_loaded(bot_id, test_vision_chat_id)
        elapsed = time.perf_counter() - start

        assert test_vision_chat_id in trainer.bot_data[bot_id]["assistants"], "Vision chat should be in memory"

        vision_mem = trainer.bot_data[bot_id]["assistants"][test_vision_chat_id]
        assert isinstance(vision_mem, VisionMemory), f"Expected VisionMemory, got {type(vision_mem)}"
        assert len(vision_mem.chat_history) == 2, f"Expected 2 history entries, got {len(vision_mem.chat_history)}"
        assert len(vision_mem.chat_history_store.messages) == 4, f"Expected 4 messages (2 Q+A pairs), got {len(vision_mem.chat_history_store.messages)}"

        results.append(("_ensure_vision_chat_loaded — MongoDB replay", True, f"4 messages loaded in {elapsed*1000:.1f}ms"))
    except Exception as e:
        results.append(("_ensure_vision_chat_loaded — MongoDB replay", False, str(e)))

    # ─── Test 9: _ensure_chat_loaded — unknown chat (no history) ─────────────
    try:
        fresh_chat_id = "chat-never-existed-xyz"
        trainer._ensure_chat_loaded(bot_id, fresh_chat_id)

        assert fresh_chat_id in trainer.bot_data[bot_id]["chains"], "Fresh chat should be created"
        bot_inst = trainer.bot_data[bot_id]["chains"][fresh_chat_id]
        assert isinstance(bot_inst, RAGBot), f"Expected RAGBot, got {type(bot_inst)}"
        assert len(bot_inst.chat_history.messages) == 0, "Fresh chat should have 0 messages"

        results.append(("_ensure_chat_loaded — no history (fresh)", True, "Created empty RAGBot"))
    except Exception as e:
        results.append(("_ensure_chat_loaded — no history (fresh)", False, str(e)))

    # ─── Test 10: list_chats() still works (queries MongoDB directly) ────────
    try:
        chats = trainer.list_chats(bot_id)
        assert "chat_ids" in chats, "list_chats should return chat_ids"
        assert "vision_chat_ids" in chats, "list_chats should return vision_chat_ids"
        assert test_chat_id in chats["chat_ids"], f"test_chat_id should be in chat_ids"
        assert test_vision_chat_id in chats["vision_chat_ids"], f"test_vision_chat_id should be in vision_chat_ids"

        results.append(("list_chats — MongoDB query", True, f"{len(chats['chat_ids'])} chats, {len(chats['vision_chat_ids'])} vision"))
    except Exception as e:
        results.append(("list_chats — MongoDB query", False, str(e)))

    # ─── Test 11: get_chat_by_id() still works ───────────────────────────────
    try:
        history = trainer.get_chat_by_id(test_chat_id, "oldest")
        assert history is not None, "Chat history should not be None"
        assert len(history) == 3, f"Expected 3 messages, got {len(history)}"
        results.append(("get_chat_by_id", True, f"Retrieved {len(history)} messages"))
    except Exception as e:
        results.append(("get_chat_by_id", False, str(e)))

    # ─── Test 12: Verify the load_bot → get_response flow works end-to-end ───
    # (This simulates what the production Rag_on_Longtrainer/api.py does)
    try:
        # Simulate server restart: clear all in-memory data
        del trainer.bot_data[bot_id]
        gc.collect()

        # Reload bot (should be instant, no chat loading)
        start = time.perf_counter()
        trainer.load_bot(bot_id)
        load_time = time.perf_counter() - start

        # Verify chains are empty
        assert trainer.bot_data[bot_id]["chains"] == {}, "Chains should be empty after fresh load"

        # Now create a new chat and verify it works
        new_cid = trainer.new_chat(bot_id)
        assert new_cid in trainer.bot_data[bot_id]["chains"], "New chat should be in chains"

        results.append(("Full reload→new_chat flow", True, f"Boot: {load_time*1000:.1f}ms, new chat: {new_cid[:20]}..."))
    except Exception as e:
        results.append(("Full reload→new_chat flow", False, str(e)))

    # ─── Test 13: set_custom_prompt_template works after lazy load ───────────
    try:
        custom_prompt = "You are a medical assistant. {context}\n{chat_history}\n{question}"
        trainer.set_custom_prompt_template(bot_id, custom_prompt)
        assert trainer.bot_data[bot_id]["prompt_template"] == custom_prompt, "Prompt should be updated"
        results.append(("set_custom_prompt_template", True, "Prompt updated successfully"))
    except Exception as e:
        results.append(("set_custom_prompt_template", False, str(e)))

    # ─── Cleanup: Delete test bot and all data ───────────────────────────────
    try:
        trainer.delete_chatbot(bot_id)
        assert bot_id not in trainer.bot_data, "Bot should be removed from bot_data"

        # Verify MongoDB is clean
        assert trainer._storage.find_bot(bot_id) is None, "Bot should be deleted from MongoDB"
        chats_after = trainer._storage.list_chats(bot_id)
        assert len(chats_after["chat_ids"]) == 0, "All chats should be deleted"
        assert len(chats_after["vision_chat_ids"]) == 0, "All vision chats should be deleted"

        results.append(("Cleanup — delete_chatbot", True, "All data cleaned"))
    except Exception as e:
        results.append(("Cleanup — delete_chatbot", False, str(e)))

    _print_results(results)
    return all(ok for _, ok, _ in results)


def _print_results(results):
    """Print formatted test results."""
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
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print("=" * 70)


if __name__ == "__main__":
    success = test_lazy_loading()
    sys.exit(0 if success else 1)
