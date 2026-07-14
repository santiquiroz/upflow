# Contributing to Upflow

Thanks for helping out. Upflow is open source (MIT) and PRs are welcome.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-realesrgan.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-ffmpeg.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\download-rife.ps1  # optional, only for FPS boost work
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\uvicorn app.main:app --reload --port 8090
```

## Ground rules

- **Atomic code**: one function, one job; the name says what it does.
- **Low cyclomatic complexity**: extract branches into named functions, prefer early returns.
- **No doc-comments**: names should be enough. Single-line comment only when the *why* isn't obvious.
- **Tests required** for new logic. Run `pytest` before opening a PR.
- **Never commit** `vendor/`, `runtime/`, `.venv/`, or `.env` (all gitignored).

## Where help is most useful

Disk cleanup, job retention and the RIFE FPS-boost stage are done. See [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md) (Phase 6) and [`docs/RESEARCH_ANIME_SUITE.md`](docs/RESEARCH_ANIME_SUITE.md) for what's next:

1. AI audio enhancement (DeepFilterNet CLI, denoise/dialogue as an optional pipeline stage).
2. AI subtitles (whisper.cpp, generation + translation, muxed as a soft track).
3. Quality/speed slider (Fast/Balanced/Best presets mapped to real per-engine knobs).
4. Batch mode for full anime seasons (multi-upload, aggregate progress).

## PR checklist

- [ ] `pytest` passes
- [ ] No secrets, no `vendor/`/`runtime/` artifacts committed
- [ ] New logic has tests
- [ ] Commit messages follow `type: description` (feat, fix, refactor, docs, test, chore)
