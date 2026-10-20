# Phase A — Test and CI Integrity

> Historical verification record: Python 3.10 results below predate the minimum-version change on 7 October 2026. Current installation requires Python 3.11+, and CI tests Python 3.11/3.12.

**Implemented:** 5 October 2026

**Baseline:** `8368268` (`1.3.1`)

**Branch:** `fix/test-ci-integrity`

**Scope:** test discovery, failure signaling, lint enforcement, and dependency setup.

## Changes

| Area | Before | After |
|---|---|---|
| CI collection | Four standalone scripts and 15 explicitly selected pytest cases. | Full `pytest tests/ -v -ra` collection on the existing Python 3.10/3.11/3.12 matrix. |
| Legacy aggregate tests | A returned `False` could pass when invoked through pytest. | The four original check groups assert their aggregate result; standalone execution still works. Returning a value from any test is treated as an error. |
| Bug-fix regressions | 17 cases outside `tests/`, omitted by CI. | Moved to `tests/test_bugfixes_pre_phase3.py`, retaining every original assertion. |
| Rate limiting | Seven cases omitted by CI's filename list. | All seven included by full collection. |
| Lazy loading | Service-dependent script omitted by CI; setup failure could return `False`. | 16 offline cases plus the retained service-backed test, whose aggregate failure raises an assertion. Two additional cases verify failure signaling. |
| Integration setup | No dedicated service job or Qdrant extra. | MongoDB 7 service, in-memory Qdrant, explicit integration dependencies, fake embeddings and an unused placeholder API key; no paid model requests. Pytest-generated database files stay in its temporary directory. |
| Structured tests | Missing `longtrainer.structured` aborts full collection. | Explicit temporary module skip linked to Phase B. Internal import errors will still fail once the module exists. |
| Lint | CI runs a small flake8 subset; configured Ruff reports 211 baseline violations. | CI runs the bounded Ruff rules below. Existing collected-test import cleanup and CLI literal cleanup are isolated in mechanical commit `59de738`. |
| Fresh installation | Community tool loading fails without `mypy_extensions`; document parsing downloads its language model during tests. | Declare `mypy-extensions>=1.0`; download the spaCy model during CI setup before offline tests. |

Application methods, routes, response contracts, and package version remain unchanged. The CLI edits only remove redundant `f` prefixes from literal strings. Agent types and structured-output extraction belong to later phases.

## Run the checks

Install the development dependencies and parser asset in an activated Python environment:

```bash
python -m pip install -e '.[agent,dev,cli,api]'
python -m pip install 'spacy>=3.8,<3.9' 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
ruff check .
pytest tests/ -v -ra
```

Document parsing also requires the existing CI system dependencies: libmagic, Poppler, and Tesseract. Installation and parser preparation require network access; the default tests use local fixtures and fake or mocked external services.

For service-backed tests, use a dedicated test MongoDB instance:

```bash
python -m pip install -e '.[agent,dev,cli,api,integration]'
export LONGTRAINER_TEST_MONGO_URI='mongodb://localhost:27017/?serverSelectionTimeoutMS=5000'
pytest tests/ -v -ra -m integration --run-integration
```

The `integration` marker skips service tests by default. The explicit CI integration job enables them; it uses a MongoDB health check and Qdrant's local in-memory implementation, so no Qdrant server is required.

## Verification evidence

Validation used an isolated Python **3.12.14** environment. Both dependency sets were installed from this checkout using `uv pip install -e`, with CPU PyTorch wheels. The environment used pytest 9.1.1 and Ruff 0.16.10. Baseline comparisons used an archive of `8368268` with the same environment.

| Check | Result |
|---|---|
| Baseline's selected pytest command | 15 passed. |
| Four standalone legacy scripts, before and after | All exit 0; respectively 9 import, 5 loader, 8 tool, and 6 architecture checks pass in both versions. |
| Baseline full collection | 27 collected; aborted with one missing-structured-module collection error, exit 2. |
| Baseline Ruff configuration | 211 violations: 69 W293, 46 F401, 37 I001, 24 E501, 15 W291, 10 F541, 8 E402, 1 E731, 1 F841. |
| Phase A full suite | **61 passed, 2 skipped**, 2 deprecation warnings; 62 test items collected plus one skipped module. |
| Service-backed lazy loading | **1 passed**, plus the structured-module skip. |
| Phase A Ruff | All selected checks pass. |
| Intentional assertion failure under the full CI command | Exit **1**. |
| Intentional import/collection failure under the full CI command | Exit **2**. |
| Intentional test returning `False` under the full CI command | Exit **1**, through `PytestReturnNotNoneWarning` promoted to an error. |

The temporary failure probes were removed after verification. The two permanent failure-signaling regressions inject a failed loader subcheck and failed integration setup, then verify those failures raise assertions.

These are test-case counts, not line-coverage percentages. Each legacy aggregate case also executes its original internal checks; those must not be counted as additional pytest cases. The 17 relocated cases, 16 offline cases, and two failure-signaling cases increase full collection from 27 to 62 items.

The two default-suite skips are deliberate: the integration case runs in its separate job, and the orphaned structured module waits for Phase B. Its skip covers **all 17 existing functions**, including four vision tests; none were deleted. Phase B must remove the skip and restore their collection. Deprecation warnings from LangChain are recorded rather than repaired in this phase.

The API TestClient timed out inside the local sandbox and passed outside it. Final verification therefore ran outside the sandbox. GitHub Actions and Python 3.10/3.11 have **not** been executed locally; their existing matrix is preserved in the workflow. The temporary MongoDB service is removed after verification.

## Enforced rules and follow-up tracker

Ruff enforces **E4, E7, E9, and F**, including syntax/control-flow mistakes, undefined names, and unused imports unless explicitly exempted below. Import sorting (`I`), whitespace rules (`W`), and line length (`E501`) are deferred to avoid broad source churn. No unsafe autofixes were applied.

The following are local follow-up issue records, **not published GitHub issues**. Publishing external issues or a PR is outside this local implementation.

| ID | Deferred work | Required completion |
|---|---|---|
| CI-A-01 | Restore `I`, `W`, and `E501`. | Review isolated mechanical changes, re-enable these rules, and pass the full suite. |
| CI-A-02 | Audit retained production `F401` imports in `bot.py`, `chat.py`, `rate_limiter.py`, `tools.py`, `trainer.py`, and `vectorstores.py`; audit `trainer.py`'s `F841`. | Check import side effects and module-level compatibility before removal; preserve the unused assignment's call side effects. Remove exceptions individually when verified. |
| CI-B-01 | Resolve the missing structured module and `bot.py`'s `E731` lambda. | Implement Phase B, remove the module skip and lambda exception, and collect all 17 structured/vision functions. |

Two exceptions are intentional rather than deferred cleanup: `F401` in the import smoke tests (`test_01` and `test_04`), where successful import is the check; and `E402` in `api.py`, where environment loading deliberately precedes imports. Do not delete those checks or reorder application initialization solely to satisfy lint.

## Phase B follow-up

Phase B resolves CI-B-01: `longtrainer.structured` now exists, the module skip and `bot.py` E731 exception are removed, and all 17 preserved functions collect. See [Phase B verification](phase_b_structured_output.md). The counts above remain the historical Phase A results.

## Remote CI parser setup follow-up — 6 October 2026

[PR 26's Python 3.10 job](https://github.com/ENDEVSOLS/Long-Trainer/actions/runs/37417949931/job/112120710490) failed in `Prepare document parser assets`, before its tests ran. The public job metadata confirms the failing step; the full log requires authenticated access. Python 3.11/3.12 were cancelled by the matrix fail-fast behavior; lint and MongoDB integration passed.

The corresponding local Python 3.10 development environment reproduced `python -m spacy download en_core_web_sm` failing with `No module named spacy`. Its resolved Unstructured 0.18.32 dependencies do not include spaCy; the English model 3.8.0 wheel also declares no dependencies. The previous CI command therefore relied on a transitive dependency that is absent in this resolution.

The workflow now explicitly installs `spacy>=3.8,<3.9` and the matching `en_core_web_sm` 3.8.0 wheel, then calls `spacy.load('en_core_web_sm')` to validate preparation. Matrix fail-fast is disabled so one failure cannot cancel the other Python results. README/current setup instructions match this change. The workflow fix is local until committed and pushed; remote confirmation remains pending.

Follow-up local verification: explicit installation selected spaCy 3.8.16; `spacy.load("en_core_web_sm")` passed on Python 3.10, 3.11 and 3.12. The post-install Python 3.10 suite passed **141 tests, 1 integration skip**, 11 existing deprecation warnings in 30.34s. Workflow YAML checks, Ruff and `git diff --check` passed. No commit or push was performed.
