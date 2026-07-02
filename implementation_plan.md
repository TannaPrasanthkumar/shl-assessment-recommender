# Implementation Plan: SHL Assessment Recommender Refactoring & Compliance

Exhaustive engineering review of the SHL Assessment Recommender platform, followed by targeted refactoring to resolve all critical architectural, retrieval, testing, and compliance issues.

## User Review Required

> [!IMPORTANT]
> The refactoring preserves the stateless nature and modular architecture of the system. It introduces dynamic, grounded catalog comparisons/explanations (replacing previous static placeholders) and strict API response timeout handling (28s turn cap limit).
> It also implements a dual-mode strategy for LLM calls:
> 1. **Offline Grounded Fallback**: Uses rule-based DB lookups and structured Markdown templates to build zero-hallucination comparison tables.
> 2. **Online LLM Mode**: Calls the Gemini API when `GEMINI_API_KEY` is provided, using a grounded prompt context.

## Proposed Changes

### Configuration & Schemas

#### [MODIFY] [pyproject.toml](file:///c:/Projects/SHL%20Research/pyproject.toml)
- Add `jinja2` to explicit dependencies to fix the transitive dependency code smell.

#### [MODIFY] [schemas.py](file:///c:/Projects/SHL%20Research/src/models/schemas.py)
- Create `HealthResponse` schema to strictly validate the `/health` endpoint output.

---

### Core Service & Engine

#### [MODIFY] [main.py](file:///c:/Projects/SHL%20Research/src/main.py)
- Use `HealthResponse` for `/health` endpoint response model.
- Replace hardcoded absolute path fallbacks with path resolution based on `settings.BASE_DIR`.
- Enforce a 28-second execution timeout on the `/chat` route using `asyncio.timeout`. If a timeout is hit, catch it and return a schema-compliant HTTP 200 response with `end_of_conversation=False`, recommendations `[]`, and a user-friendly timeout message.

#### [MODIFY] [vector_store.py](file:///c:/Projects/SHL%20Research/src/database/vector_store.py)
- Add an `asyncio.Lock()` to `SHLCatalogRepository` to prevent concurrent/duplicate catalog loading race conditions.
- Replace hardcoded absolute path strings with dynamic path resolving.

#### [MODIFY] [decision_engine.py](file:///c:/Projects/SHL%20Research/src/agents/decision_engine.py)
- Replace static dummy text for `COMPARE` and `EXPLAIN` actions.
- Implement an offline-friendly catalog matcher that identifies mentioned assessments (e.g. OPQ, G+, Verify) in the query, queries the catalog database, and generates a detailed, grounded side-by-side Markdown table comparison or explanation.
- Integrate optional Gemini API calling if `GEMINI_API_KEY` is present.

---

## Verification Plan

### Automated Tests
- Run `pytest` inside the virtual environment:
  ```powershell
  .\myenv\Scripts\pytest
  ```
- Run the simulation evaluator script:
  ```powershell
  .\myenv\Scripts\python -m tests.evaluator_simulator
  ```

### Manual Verification
- Perform API validation on `/health` and `/chat` to verify schema formats:
  ```powershell
  Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
  ```
