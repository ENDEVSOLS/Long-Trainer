# Live tool and Gemini verification — 2026-10-06

## Verified changes

- Bound the research extra to `arxiv>=2.1,<4`. The community wrapper calls
  `Search.results`, which arxiv 4 removed. A regression invokes the real wrapper
  with the installed client API and only substitutes the remote result iterator.
  It failed with arxiv 4.0.1, passes with arxiv 3.0.0, and live paper lookup passes.
  Upstream documents the removal in its
  [4.0.0 release notes](https://github.com/lukasschwab/arxiv.py/releases/tag/4.0.0).
- Configure a descriptive project user agent when loading Wikipedia tools.
  Default requests initially received HTTP 403 and failed JSON decoding;
  live retrieval through the updated factory now succeeds.
- Preserve Yahoo's existing article response and fall back to yfinance's news
  search when ticker news returns no articles. The fallback returns actual
  headlines, publishers, UTC publication timestamps, and source links, explicitly
  identifying that full article text is unavailable. Empty search results retain
  the original no-news response. Live MSFT news and a Gemini financial-agent
  tool call both pass.
- Extract text from provider content blocks for structured responses, agent
  responses, sync/async agent streaming, and vision responses. Live Gemini
  revealed list/string failures; regressions reproduced them before the fix.
- Install all purpose extras in the offline CI matrix so optional compatibility
  regressions are exercised without making credentialed network calls in CI.

## Project checks

The complete source suite ran with real temporary MongoDB and the Qdrant
integration extra. These are distinct from the live Gemini checks below.

| Environment | Result |
| --- | --- |
| Python 3.10.21 | 151 passed, no skips |
| Python 3.11.16 | 151 passed, no skips |
| Python 3.12.14 | 151 passed, no skips |
| Ruff | Passed |
| Strict MkDocs build | Passed |
| Whitespace/diff checks | Passed |
| Dependency consistency, integration and Gemini environments | Passed |
| Wheel build | Passed; version 1.4.0, research requires arxiv below 4 |
| Fresh wheel installation, Python 3.12 | Passed: all purpose extras, 151 tests with integration, no skips |

Each source suite reported 12 deprecation warnings from dependencies, including
LangGraph's existing agent factory and arxiv's legacy method. They did not fail
checks. The arxiv upper bound retains compatibility until the community wrapper
uses the current client API.

Logs are local temporary artifacts:
`/tmp/longtrainer-final-310.log`, `/tmp/longtrainer-final-311.log`,
`/tmp/longtrainer-final-312.log`.

The fresh wheel suite ran from `/tmp`, confirmed package imports from the new
environment's `site-packages`, and resolved arxiv 3.0.0. It used CPU PyTorch
wheels and installed the spaCy parser model required by loader tests. The log is
`/tmp/longtrainer-final-wheel.log`. An initial attempt using pytest's importlib
mode could not resolve legacy cross-test imports; the project’s normal pytest
import mode passed without changing test semantics.

## Live service checks

Gemini used the supplied test credentials from the outer workspace `.env` with
model `gemini-3.1-flash-lite`. Credentials were not printed, copied into this
report, or included in the wheel.

| Check | Result |
| --- | --- |
| DuckDuckGo web search | Passed: actual results, snippets, and links |
| Wikipedia | Passed: actual Python article summary |
| arXiv | Passed: actual paper `1706.03762`, “Attention Is All You Need” |
| Python REPL | Passed: actual trusted arithmetic execution requested by Gemini |
| Document reader | Passed: actual text-file ingestion |
| Yahoo Finance News | Passed: actual MSFT headlines with publishers, timestamps, and links |
| Tavily | Not verified: no `TAVILY_API_KEY` configured |
| Gemini direct response | Passed |
| Gemini RAG response and streaming | Passed |
| Gemini JSON-schema response | Passed: integer warranty duration 24 |
| Gemini agent tool call | Passed: actual Python invocation, result 437 |
| Gemini sync/async agent streaming | Passed: arithmetic results 56 and 72 |
| Gemini vision through VisionBot | Passed: generated red JPEG recognized, string-compatible response returned |
| Customer-support named bot | Passed: actual file ingestion, FAISS, MongoDB, and live Gemini |
| Saved customer-support bot/chat reload | Passed with live Gemini |
| Structured trainer response and persistence | Passed with actual MongoDB |
| Research named bot | Passed with explicit arxiv tool override |
| Coding named bot | Passed with explicit Python tool override |
| Financial named bot | Passed arithmetic with explicit Python tool override |
| Financial default tools | Passed: callback confirmed `yahoo_finance_news` invocation; Gemini cited actual news links and publication timestamps |

The financial arithmetic assertion was corrected to accept a thousands separator
in the captured `1,050` response. This was a test assertion issue, not a model or
project failure.

One initial research-agent conversation stalled; a bounded retry passed. Live
provider/service calls are subject to latency and availability; successful smoke
checks do not establish reliability under all network conditions.

## Scope and remaining limits

- Full default research/coding conversations still require live Tavily
  verification. Their explicit-tool variants passed; those overrides are not
  evidence that Tavily works.
- FAISS/Mongo end-to-end conversations used deterministic local test embeddings.
  No external embedding-provider request was verified.
- Existing API and CLI coverage comes from the automated suite. These live
  checks exercise the model factory, bot classes, and trainer, not a deployed
  HTTP service or all CLI commands with Gemini.
- Other paid LLM providers, private cloud/document-source credentials, optional
  tracing backends, exhaustive loader formats, and load/stress behavior were not
  independently verified live. This report is not a claim that every possible
  configuration of the project works.
- These changes are local. Remote CI must run on the newly pushed commit;
  earlier green PR checks do not cover these new fixes.

Live harnesses and redacted result files remain under `/tmp/longtrainer_*live*`;
none require embedding credentials in source code. The temporary MongoDB service
is removed after verification.
