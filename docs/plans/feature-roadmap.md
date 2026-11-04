# Feature roadmap: local AI, auto-publish, UI

Status: draft for discussion (2026-11-03). Nothing implemented yet.

## Where the repo is today

| Area | Current state | Key files |
|---|---|---|
| Visual sources | 14 sources in `task.py:103`: stock (pexels, pixabay, coverr), local files, cloud gen (wavespeed, seedance, ofox, muapi, metaso_minimax, infistar, jalapeno, loomloom), image gen (`openai_image`, `codex_image`). Each source is an `if params.video_source == ...` branch in `task.py` plus a `_download_videos_<x>_on_demand` in `material.py` (3.3k lines). | `app/services/task.py`, `app/services/material.py` |
| Local image gen | Exists, but as a manual sidecar in `docs/local-image-server/`: FastAPI server exposing OpenAI `/v1/images/generations`, one hard-coded SDXL model (RealVisXL V5 Lightning), GPU hand-off with Ollama. User must create a venv, install torch, run it, and edit `config.toml` by hand. | `docs/local-image-server/server.py`, `policy.py` |
| Local video gen | None. | |
| Image → clip | `_render_openai_image_video` turns one still into a clip with `ImageMotion` (Ken Burns). | `material.py:1909` |
| Publishing | Only via Upload-Post (paid 3rd-party aggregator). One-shot after render, background thread pool, platform list, YouTube made-for-kids flag. No scheduling, no per-platform captions/hashtags, no thumbnail, no publish history UI. | `app/services/upload_post.py`, `task.py:50-230, 1178+` |
| WebUI | Streamlit, single `webui/Main.py` of 9,506 lines + `styles.css`. Settings in a dialog. | `webui/Main.py` |
| API | `/v1/videos`, `/subtitle`, `/audio`, tasks list/get/delete, stream/download, LLM script/terms. No settings or publish endpoints. | `app/controllers/v1/*.py` |

Dev box has RTX 4070 Ti SUPER 16 GB + RTX 4060 8 GB, so multi-GPU must be handled.

---

## 1. Local AI for image and video generation

### Goal
One click in the UI: detect hardware → recommend the best image and video models that fit → download → generate scenes locally. No manual venv, no editing `config.toml`.

### Design

**Sidecar, not in-process.** Keep torch/diffusers out of the main app (it would add ~5 GB of deps to every install and fight faster-whisper for VRAM). Promote `docs/local-image-server/` into a managed service:

```
local_ai/                      # new top-level package, own venv / own Docker image
  server.py                    # FastAPI: /v1/images/generations (exists), /v1/videos/generations (new), /hardware, /models, /models/{id}/download
  hardware.py                  # GPU vendor, VRAM per device, RAM, free disk, CUDA/ROCm/MPS/DirectML, driver version
  catalog.py + catalog.json    # model tiers; JSON fetched from repo raw URL so "best" can update without a release
  downloader.py                # huggingface_hub snapshot_download, resume, progress, disk precheck, HF token for gated repos
  pipelines/image_sdxl.py, image_flux.py, video_ltx.py, video_wan.py
  gpu_policy.py                # from policy.py: Ollama unload, idle offload, one job per GPU
app/services/local_ai.py       # main-app client: start/stop sidecar, health, call endpoints
```

The main app launches the sidecar as a subprocess (`app/launcher.py` already starts processes) using a venv it creates on first use with `uv`. Docker: separate `Dockerfile.local-ai` on a current CUDA base (existing `Dockerfile.gpu` is on CUDA 12.1, too old for Blackwell cards).

**Hardware → model selection.** Score = fits in VRAM (with offload headroom) × quality rank × speed. Initial catalog (verify licenses and current versions at implementation time; the list must live in `catalog.json`, not code):

| VRAM | Image | Video (text/image → video) |
|---|---|---|
| CPU only / < 6 GB | SD-Turbo / SDXL-Turbo (OpenVINO on CPU) | disabled; fall back to image + Ken Burns |
| 6–8 GB | SDXL Lightning (RealVisXL, current default) | Wan 1.3B-class / LTX-Video small, low res, CPU offload |
| 12–16 GB | FLUX schnell (fp8/GGUF) or SDXL | LTX-Video, Wan 5B-class at 480p–720p |
| 24 GB+ | FLUX dev / SD3.5 Large | Wan 14B-class / HunyuanVideo-class, 720p |

Rules: prefer image-to-video (generate keyframe with image model, animate it) — better consistency per scene than pure T2V. Flag non-commercial licenses (e.g. FLUX dev) in the UI and require opt-in. With multiple GPUs, pin each job to the biggest free card; never split one model across cards in v1.

**Integration with the pipeline.** Two new sources, modeled on `openai_image`:
- `local_image`: thin wrapper over the existing `openai_image` path pointed at the sidecar (zero new render code).
- `local_video`: new `_download_videos_local_video_on_demand` in `material.py`; per script segment → keyframe → I2V clip of `video_clip_duration`; on timeout or OOM, fall back to image + Ken Burns for that scene instead of failing the task.

Add both to the source list in `task.py:103` and the `stop_at` branches around `task.py:1594-1661`. Report per-scene progress through `generation_progress.py`; show a time estimate before starting (video gen is minutes per clip on consumer cards).

**UI.** New settings tab "Local AI": hardware card, recommended models with size/VRAM/license, download/delete with progress, disk usage, sidecar status/logs, "test generate" button.

### Phases
1. Move sidecar into `local_ai/`, add `hardware.py`, `/hardware` + `/models` endpoints, managed start from the app. Ship `local_image` with the existing SDXL model. **Done 2026-11-04**, except a `/hardware` endpoint: the app detects hardware itself, before the runtime exists.
2. Catalog + downloader + model picker UI; add FLUX tier. **Done 2026-11-04**: catalog has FLUX.1 schnell (gated), RealVisXL V5 Lightning and DreamShaper 8; downloads go to plain folders because the Hugging Face cache needs symlinks that Windows refuses without Developer Mode.
3. `local_video` with one model family (LTX-Video first: fastest on 12–16 GB), keyframe → I2V, per-scene fallback. **Done 2026-11-04 with Wan 2.2 TI2V 5B instead** (Apache-2.0; LTX-Video has a custom license and 48–93 GB repos). Loaded in 4-bit NF4 (bitsandbytes) and resharded on first load into 1 GB bf16 files, because loading 5 GB fp32 shards failed with Windows os error 1455 when WSL held most of the commit limit. About 2 minutes per 3 s 480p clip on an RTX 4070 Ti SUPER, so a 50 s video takes roughly 40 minutes; Wan 2.1 T2V 1.3B covers 8 GB cards.
4. Second video family (Wan), multi-GPU scheduling, Docker image.

### Risks
- Windows torch/CUDA install is the #1 support burden → `uv` + pinned wheel index per detected CUDA version; clear error text.
- VRAM contention with Ollama LLM and faster-whisper → reuse `policy.py` hand-off, extend to whisper.
- Gated HF repos need a token; licenses vary → surface in UI, never auto-accept.
- Disk: video models are 10–60 GB → precheck and show before download.

---

## 2. Auto-publish to YouTube, Instagram, TikTok

### Problems with today
Hard dependency on one paid aggregator; publish only immediately after render; same title for all platforms; no hashtags/description/thumbnail generation; no schedule, retry visibility, or history.

### Design

```
app/services/publishing/
  base.py          # Publisher protocol: validate(video, meta) -> issues; publish(job) -> PlatformResult; status(job)
  upload_post.py   # adapter over existing UploadPostService (keep as the easy path)
  youtube.py       # YouTube Data API v3, resumable upload, OAuth installed-app flow
  tiktok.py        # TikTok Content Posting API
  instagram.py     # Instagram Graph API (Reels)
  metadata.py      # LLM: per-platform title/caption/hashtags/description from script (reuse llm.py)
  thumbnail.py     # best frame pick, or local image model from feature 1
  scheduler.py     # persistent job queue, scheduled_at, retries w/ backoff, idempotency key
  tokens.py        # encrypted OAuth token store (pattern from chatgpt_oauth.py / codex_oauth.py)
app/controllers/v1/publish.py   # POST /v1/publish, GET /v1/publish/jobs, DELETE /v1/publish/jobs/{id}
```

**Platform reality check (drives the order):**
- **YouTube direct:** easiest. OAuth with user's own Google Cloud project; default quota allows ~6 uploads/day (1,600 units each). Shorts = vertical ≤ 3 min. Supports scheduled publish (`publishAt`), custom thumbnail, made-for-kids flag. Do first.
- **TikTok direct:** Content Posting API needs a registered app; unaudited apps can only post as private (`SELF_ONLY`). Public posting needs TikTok's audit. Keep Upload-Post as the default for TikTok until a user has an audited app.
- **Instagram direct:** needs Business/Creator account linked to a Facebook Page and a Meta app; Reels upload needs the video at a public URL (or the resumable upload endpoint). Hardest to self-host → Upload-Post default, direct as opt-in.

**Scheduling.** Jobs persisted through existing `state.py` (memory/redis). Per-account posting slots ("weekdays 18:00 local"), queue view, pause, retry. Survives restart; reuse the owner/heartbeat logic already in `task.py` (`_is_cross_post_owner_alive`) so two processes never double-post.

**Metadata.** One LLM call per video produces `{youtube: {title, description, tags}, tiktok: {caption, hashtags}, instagram: {caption, hashtags}}`, editable in the UI before publish or auto-applied in auto mode. Enforce platform limits (title 100 chars, caption 2,200, hashtag counts).

**Batch / autopilot mode.** Topic list or RSS → generate N videos → auto-schedule across slots. Optional "review before publish" gate.

### Phases
1. Extract `Publisher` protocol; wrap existing Upload-Post as adapter (no behavior change, keep tests green).
2. Metadata + thumbnail generation, editable publish form, publish history panel.
3. Scheduler + `/v1/publish` API.
4. YouTube direct.
5. TikTok and Instagram direct (opt-in, documented app setup).
6. Autopilot batch mode.

### Risks
- Platform ToS on automated posting and AI-content disclosure labels → add "AI-generated" disclosure toggles per platform (YouTube `containsSyntheticMedia`, TikTok AIGC label).
- Token leakage → encrypt at rest, never log, exclude from key-backup export unless explicit.

---

## 3. Better UI

### Problem
9.5k-line `Main.py` makes every change risky and slow; all options on one long page; settings dialog crowded; features 1 and 2 add two big panels.

### Options

| | A. Restructure Streamlit | B. New SPA (React + Vite + shadcn/ui) on FastAPI |
|---|---|---|
| Effort | ~2–3 weeks | ~6–10 weeks |
| Result | Cleaner, wizard flow, same tech | Real app feel: live preview, drag-reorder scenes, timeline editor, responsive |
| Prereq | none | API parity: settings, config, voices list, fonts, BGM, publish, progress via SSE/WebSocket |
| Risk | Streamlit limits (reruns, layout control) remain | two UIs to maintain during migration |

**Decision (2026-11-03): A — restructure Streamlit.** B is not planned; keep it possible by building new features API-first.
1. Split `Main.py` into `webui/common.py` and `webui/components/` (one module per panel). Pure move, no behavior change, existing UI tests guard it. **Done 2026-11-03** (uncommitted). Page-level modules come with step A.2 via `st.navigation`; avoid a `webui/pages/` folder, which Streamlit turns into automatic multipage navigation.
2. Create page becomes a stepper: Script → Voice → Visuals → Subtitles & Music → Render → Publish, with a sticky summary/preview column.
3. Library page: grid of rendered videos with thumbnail, state, re-run, publish status.
4. Visual pass on `styles.css`: one token set, dark/light, consistent spacing.
5. In parallel, every new feature (local AI, publish) gets API endpoints first, UI second → B becomes a front-end-only project later.

---

## Suggested order

| Step | Work | Why this order |
|---|---|---|
| 0 | Split `Main.py` into pages/components (UI phase A.1) | Unblocks UI for features 1 & 2, low risk |
| 1 | Local AI phase 1–2 (managed sidecar, hardware detect, model picker, local images) | Builds on working code, fast win |
| 2 | Publishing phase 1–3 (protocol, metadata, scheduler, history) | Value for every user, no new accounts needed |
| 3 | Local video (phase 3) | Biggest, most hardware-dependent |
| 4 | YouTube direct, then TikTok/IG direct | Gated by platform app reviews |
| 5 | UI phase A.2–A.4 (stepper, library, visual pass) | |

## Open questions
1. ~~UI: Streamlit (A) or SPA (B)?~~ Decided: A.
2. Target hardware: mainly NVIDIA on Windows, or must AMD / Apple Silicon / CPU-only work too?
3. Commercial use? Decides which model licenses are allowed by default.
4. Keep Upload-Post as default publisher, or go direct-first?
5. Must local AI work inside Docker from day one?
