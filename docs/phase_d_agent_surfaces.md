# Phase D implementation and verification

Implemented locally on `feat/agent-types-surfaces`, retaining the uncommitted Phase B/C changes. Phase D adds purpose selection to the existing HTTP build route and CLI, registers the agent guide, and completes local docs/example verification. No commit, push, remote CI run or publication was performed.

## Adopted interface decisions

- `CreateBotRequest.agent_type` is an optional string. The HTTP route validates names before service access and forwards a supplied type to `create_bot`. Unknown names return HTTP 400 with supported names. Legacy payloads keep their previous arguments and response shape; normal malformed-field validation remains HTTP 422.
- `longtrainer build [BOT_ID]` creates and builds a new empty bot when no ID is supplied, and loads/builds an existing ID otherwise. `--agent-type` is also supported on `bot create`. Existing `--agent`, `--tools`, `--prompt`, config options and command shapes remain available.
- A plain existing-ID CLI build preserves the stored named configuration. Prompt-only builds override its prompt. Tool-only builds retain the named purpose and stored prompt. Explicit purpose selection chooses that type's defaults. Explicit `--tools ""` selects no tools; an omitted tools option selects defaults for a named type. The type determines mode when both mode and type are supplied.
- Purpose, prompt and per-bot tool configuration are excluded from YAML. YAML continues to provide infrastructure settings; the sample now explicitly states this restriction. Explicit Python/API/CLI arguments and plain CLI rebuilds of persisted configuration determine behavior.
- SQL remains deferred and no SQL URI interface is introduced.

## Files and documentation

Phase D edits `longtrainer/api.py`, `longtrainer/cli.py`, `tests/test_12_agent_surfaces.py`, `cli-test.yaml`, `CHANGELOG.md`, `docs/mkdocs.yml`, `docs/docs/cli_api.md`, `docs/docs/index.md`, the agent guide and the implementation plan.

The canonical guide is now `docs/docs/agent_types.md`, which matches MkDocs' source directory. `docs/agent_types.md` retains a link for existing readers and verification records. The duplicate migration navigation entries were consolidated into one Migration Guide entry. Strict docs verification found three pre-existing references to removed GIF assets in the home page; those broken references were removed. The home page links to the upcoming 1.4 purpose interfaces without changing release versions.

Generated docs were built only under `/tmp/longtrainer-phase-d-docs-site`; tracked `docs/site/` files were not edited. Package versions remain unchanged for release preparation.

## Executed verification — 6 October 2026

| Check | Result |
| --- | --- |
| Python 3.12 offline suite | **141 passed, 1 skipped**, 11 deprecation warnings, 38.06s. |
| Python 3.12 complete suite with temporary MongoDB and in-memory Qdrant | **142 passed**, no skips, 11 warnings, 47.39s. |
| Python 3.10.21 offline suite | **141 passed, 1 skipped**, 11 warnings, 34.43s. |
| Python 3.11.16 offline suite | **141 passed, 1 skipped**, 11 warnings, 39.35s. |
| Strict MkDocs build | Passed, generated site under `/tmp`; no MkDocs missing-link warnings. |
| CLI help | `longtrainer --help`, `longtrainer build --help`, and `longtrainer bot create --help` all exit 0. |
| Documented research example | `build --agent-type research` constructs all three real default tools, persists the configuration to real MongoDB and responds through the existing agent runtime with a local fake model. Existing-ID `build ... --agent-type coding --tools ""` also passes. |
| Ruff and patch whitespace | `ruff check .` and `git diff --check` pass. |

The 25 new test cases cover all four HTTP purpose names, HTTP 400 errors before storage access, legacy payloads, dependency failures, both CLI creation commands, absent/list/empty tool overrides, invalid names before allocation, persisted rebuilds, legacy CLI options, visible creation failures and YAML exclusion. Six cases run the API/CLI paths through real trainer/runtime construction using fake models and controlled storage/retrieval fixtures; they assert persisted mode, RAG/Agent class and actual response behavior.

The Python 3.10/3.11 offline skip is the existing service-dependent integration test, which passed in the complete Python 3.12 run. Existing LangChain/LangGraph deprecations remain. Material emits its own upstream informational banner, but strict MkDocs has no build warnings after the broken-link repair.

Commands ran from `Long-Trainer/`:

```bash
/tmp/longtrainer-phase-a-venv/bin/pytest tests/ -q -ra
/tmp/longtrainer-c-py310/bin/pytest tests/ -q -ra
/tmp/longtrainer-c-py311/bin/pytest tests/ -q -ra

docker run --detach --name longtrainer-phase-d-verification-mongo \
  --publish 127.0.0.1:27019:27017 docker.io/library/mongo:7
LONGTRAINER_TEST_MONGO_URI='mongodb://127.0.0.1:27019/?serverSelectionTimeoutMS=5000' \
  /tmp/longtrainer-phase-a-venv/bin/pytest tests/ -q -ra --run-integration

/tmp/longtrainer-phase-a-venv/bin/mkdocs build --strict \
  --config-file docs/mkdocs.yml --site-dir /tmp/longtrainer-phase-d-docs-site
/tmp/longtrainer-phase-a-venv/bin/longtrainer --help
/tmp/longtrainer-phase-a-venv/bin/longtrainer build --help
/tmp/longtrainer-phase-a-venv/bin/longtrainer bot create --help
/tmp/longtrainer-phase-a-venv/bin/ruff check .
git diff --check
```

MkDocs/Material were installed in the existing temporary development environment. The first installation approval review timed out; one retry succeeded. An initial full-suite run exposed the missing preserved prompt on a tool-only existing-bot build; the final implementation preserves it and the full suite passes. The initial strict build failed on the three removed GIF references; the final strict build passes.

The research example ran from `/tmp/longtrainer-phase-d-research-example.py` using `/tmp/longtrainer-c-purpose-extras/bin/python`, with real default tool factories, local fake model/embeddings, a temporary filesystem directory and the dedicated MongoDB container. CLI trainer construction was supplied the controlled local model rather than an external model credential. The placeholder Tavily key was used only for constructor validation; no live search, financial data request or paid model invocation was made. Example bot records and the temporary directory were cleaned up; the dedicated MongoDB container is removed after verification.

Logs are retained under `/tmp/longtrainer-phase-d-*.log`. All Phase D local implementation/acceptance checks are satisfied. Remote CI, human review and final release/version/publication remain pending; live provider behavior is not claimed.
