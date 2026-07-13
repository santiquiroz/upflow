# Contributing to Upflow

Thanks for helping out. Upflow is open source (MIT) and PRs are welcome.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-realesrgan.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-ffmpeg.ps1
.\.venv\Scripts\uvicorn app.main:app --reload --port 8090
```

## Ground rules

- **Atomic code**: one function, one job; the name says what it does.
- **Low cyclomatic complexity**: extract branches into named functions, prefer early returns.
- **No doc-comments**: names should be enough. Single-line comment only when the *why* isn't obvious.
- **Tests required** for new logic. Run `pytest` before opening a PR.
- **Never commit** `vendor/`, `runtime/`, `.venv/`, or `.env` (all gitignored).

## Where help is most useful

See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md). High-value areas right now:

1. Disk cleanup + job retention (the `runtime/` folder currently grows without bound).
2. RIFE frame-interpolation stage (the headline roadmap feature).
3. Test coverage — only a health-check test exists today.

## PR checklist

- [ ] `pytest` passes
- [ ] No secrets, no `vendor/`/`runtime/` artifacts committed
- [ ] New logic has tests
- [ ] Commit messages follow `type: description` (feat, fix, refactor, docs, test, chore)
