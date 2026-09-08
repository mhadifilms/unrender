# Development instructions

- Keep original media immutable and generated artifacts outside the checkout.
- Work on a feature branch. Make changes proportional to the requested outcome.
- Core supports AudioShake separation and human-labeled face and voice groups.
- Dubbing, dialogue replacement, speech synthesis, model experiments, automatic speaker-to-face assignment, and production integrations are separate projects.
- Keep examples generic; never commit credentials, private endpoints, host paths, or client media.
- Install from source commits; do not add release tags or package publishing workflows.
- Run `ruff check .`, `black --check .`, `pytest -q`, and `mypy` for package changes.
- Optional model integrations must import lazily so the base CLI works without model weights.
