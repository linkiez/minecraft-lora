# Contributing

Thanks for helping. This is a small project, so the process is light.

## Set up

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests
.venv/bin/python -m ruff check scripts tests
```

The tests need no GPU, ComfyUI, models, or game assets. If you add a feature or fix a bug, add a test that fails
without your change.

## Guidelines

- Code, comments, docstrings, and documentation are in English.
- Keep each source file under 500 lines. Split larger changes into modules.
- When you add or change a command-line option, update [docs/configuration.md](docs/configuration.md). A test checks
  that every option written in the docs exists in the scripts.
- Do not commit game textures, mod assets, resource packs, models, or LoRA weights. The data folders are ignored by git
  for that reason.
- Check generation changes by looking at real outputs (a source-versus-result contact sheet), not only the tests.

## Pull requests

Keep a pull request to one logical change. Use [Conventional Commits](https://www.conventionalcommits.org/) for
messages (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`), and describe what changed and why.

## Reporting problems

Open an issue with the command you ran, the relevant part of the output, your GPU and ComfyUI version, and, for
quality problems, the source texture and the result (if you are allowed to share them).
