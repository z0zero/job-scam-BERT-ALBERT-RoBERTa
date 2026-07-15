# Repository Guidelines

## Project Structure & Module Organization

`app.py` starts Streamlit. `src/` follows MVC: controllers coordinate flows, models own ML/auth/session/Supabase logic, and views render UI; `tests/` mirrors it. Root notebooks include `research_pipeline.ipynb` and `eda_text_visualization.ipynb`; `scripts/validate_research_notebook.py` checks structure. Tracked results live under `artifacts/figures/` and `artifacts/summary/`. `best_model/`, training output, the dataset, logs, and local secrets are ignored. Supabase migrations/tests live under `supabase/`; designs and plans under `docs/superpowers/`.

## Build, Test, and Development Commands

Use CPython 3.12: pinned Torch wheels target 3.12, while the Python 3.11 devcontainer cannot install them as configured.

- `python -m venv venv`; activate with `venv\Scripts\activate` or `source venv/bin/activate`, then run `pip install -r requirements.txt`.
- `streamlit run app.py` starts the app; image OCR also requires `tesseract-ocr` from `packages.txt`.
- `python -m unittest discover -s tests -v` runs all unit tests. Target a module with `python -m unittest tests.models.test_auth_service -v`.
- `python scripts/validate_research_notebook.py` validates notebook headings/markers without executing cells.

No build, CI, formatter, linter, type-checker, or coverage configuration exists.

## Coding Style & Testing Guidelines

Use four spaces, grouped standard/third-party/local imports, `snake_case` modules/functions/tests, `PascalCase` classes, and `UPPER_SNAKE_CASE` constants. Follow nearby typing and frozen-dataclass patterns. Chain provider failures into user-safe domain exceptions without exposing internals. Tests use `unittest`, `unittest.mock`, and injected fakes; mirror source packages, use `test_*.py`/`test_*`, and add regressions with behavior changes.

## Commit & Pull Request Guidelines

Most history uses `<type>: <lowercase imperative summary>` with `feat`, `fix`, `test`, `docs`, `chore`, or `refactor`; scopes are not observed. Keep commits single-purpose. With no PR template, include scope, linked issues when applicable, exact verification results, UI screenshots for view changes, and migration/RLS evidence for Supabase changes. Flag notebook or model-artifact changes.

## Security & Configuration

Copy `.streamlit/secrets.example.toml` to ignored `.streamlit/secrets.toml`; never commit credentials or use service-role keys. Keep redirects allowlisted and provider errors generic. Test migrations in development, run `supabase/tests/analysis_history_rls.sql` with two verified users, then check Supabase advisors. Never deploy the devcontainer's disabled CORS/XSRF flags.

## Agent Workflow & Skill Routing

Use Superpowers as the sole lifecycle authority. Route with `superpowers:using-superpowers`; as triggered, use brainstorming, worktree, planning, execution, TDD, debugging, review, verification, and branch-completion owners shown below. Always use `superpowers:verification-before-completion` before completion.

Select one workflow owner, zero or more non-overlapping `agent-skills` specialists, and one explicit verification path. Use specialists such as `agent-skills:security-and-hardening`, `agent-skills:frontend-ui-engineering`, or `agent-skills:performance-optimization` only when their domain is required. Do not invoke every skill mechanically.

| Responsibility | Superpowers owner | Do not combine with |
| --- | --- | --- |
| Design | `superpowers:brainstorming` | `agent-skills:interview-me`, `agent-skills:idea-refine`, `agent-skills:spec-driven-development` |
| Planning | `superpowers:writing-plans` | `agent-skills:planning-and-task-breakdown` |
| Execution | `superpowers:subagent-driven-development` or `superpowers:executing-plans` | `agent-skills:incremental-implementation` |
| TDD | `superpowers:test-driven-development` | `agent-skills:test-driven-development` |
| Debugging | `superpowers:systematic-debugging` | `agent-skills:debugging-and-error-recovery` |
| Review | `superpowers:requesting-code-review`, `superpowers:receiving-code-review` | `agent-skills:code-review-and-quality` |
| Git isolation/completion | `superpowers:using-git-worktrees`, `superpowers:finishing-a-development-branch` | `agent-skills:git-workflow-and-versioning` |
| Routing | `superpowers:using-superpowers` | `agent-skills:using-agent-skills` |

Precedence: explicit user instructions, nearest `AGENTS.md`, Superpowers lifecycle, Agent Skills specialist guidance, then general defaults.
