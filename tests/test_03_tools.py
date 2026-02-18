"""Test 3: Tool Registry & Built-in Tools

Tests the ToolRegistry class and built-in tools WITHOUT any LLM.
"""

import sys


def test_tool_registry():
    """Test tool registration, listing, and removal."""
    print("=" * 60)
    print("TEST 3: Tool Registry & Built-in Tools")
    print("=" * 60)

    from langchain_core.tools import tool, BaseTool
    from longtrainer.tools import ToolRegistry, web_search, document_reader, get_builtin_tools

    results = []

    # 1. ToolRegistry basics
    try:
        registry = ToolRegistry()
        assert len(registry.get_tools()) == 0, "Registry should start empty"
        results.append(("ToolRegistry init (empty)", True, ""))
    except Exception as e:
        results.append(("ToolRegistry init", False, str(e)))

    # 2. Register built-in tools
    try:
        registry.register(web_search)
        registry.register(document_reader)
        assert len(registry.get_tools()) == 2
        assert "web_search" in registry.list_tool_names()
        assert "document_reader" in registry.list_tool_names()
        results.append(("Register built-in tools", True, f"tools: {registry.list_tool_names()}"))
    except Exception as e:
        results.append(("Register built-in tools", False, str(e)))

    # 3. Get tool by name
    try:
        t = registry.get("web_search")
        assert t is not None
        assert t.name == "web_search"
        results.append(("Get tool by name", True, ""))
    except Exception as e:
        results.append(("Get tool by name", False, str(e)))

    # 4. Unregister tool
    try:
        registry.unregister("document_reader")
        assert len(registry.get_tools()) == 1
        assert "document_reader" not in registry.list_tool_names()
        results.append(("Unregister tool", True, f"remaining: {registry.list_tool_names()}"))
    except Exception as e:
        results.append(("Unregister tool", False, str(e)))

    # 5. Register custom tool
    try:
        @tool
        def calculator(expression: str) -> str:
            """Evaluate a math expression and return the result."""
            return str(eval(expression))

        registry.register(calculator)
        assert "calculator" in registry.list_tool_names()
        results.append(("Register custom @tool", True, f"tools: {registry.list_tool_names()}"))
    except Exception as e:
        results.append(("Register custom @tool", False, str(e)))

    # 6. Custom tool invocation
    try:
        calc = registry.get("calculator")
        result = calc.invoke("2 + 3 * 4")
        assert result == "14", f"Expected '14', got '{result}'"
        results.append(("Custom tool invocation", True, f"2+3*4 = {result}"))
    except Exception as e:
        results.append(("Custom tool invocation", False, str(e)))

    # 7. get_builtin_tools()
    try:
        builtins = get_builtin_tools()
        assert len(builtins) == 2
        names = [t.name for t in builtins]
        assert "web_search" in names
        assert "document_reader" in names
        results.append(("get_builtin_tools()", True, f"{names}"))
    except Exception as e:
        results.append(("get_builtin_tools()", False, str(e)))

    # 8. Duplicate registration raises
    try:
        registry2 = ToolRegistry()
        registry2.register(web_search)
        try:
            registry2.register(web_search)
            results.append(("Duplicate registration guard", False, "Should have raised ValueError"))
        except ValueError:
            results.append(("Duplicate registration guard", True, "ValueError raised as expected"))
    except Exception as e:
        results.append(("Duplicate registration guard", False, str(e)))

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
    success = test_tool_registry()
    sys.exit(0 if success else 1)
