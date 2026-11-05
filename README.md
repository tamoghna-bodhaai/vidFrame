# vidFrame

**Multimodal study notes from video.** vidFrame independently reads sampled visuals and speech,
keeps timestamped evidence for both, and compiles source-grounded study notes with equations,
canonical diagrams, Markdown/JSON, and PDF export.

- OpenCV frame extraction with perceptual near-duplicate skipping and automatic scene segmentation
- Local OpenAI-compatible vision backends (**vLLM**, **Ollama**, mocked) plus hosted **OpenRouter**
  models, selected per job or in Settings
- Structured per-frame observations plus a two-stage compiled Markdown/JSON report
- Real-time live notes while frames are analyzed, with per-frame progress events
- VLM-only structured diagram/equation extraction, timestamp-aligned transcript artifacts, and
  source crops for provenance
- Controlled LaTex PDF export with safe equation fallback and optional transcript appendix
- Web UI with live progress, visual timeline, searchable frame browser, and lightbox
- Recompile from stored observations, re-run vision analysis, or reprocess with a new sample rate
- Every report version is kept on disk for comparison

---

## Quick start

Requirements: Python 3.10+, **FFmpeg** (audio extraction), **pdflatex** (PDF export), and a video file. `run.sh` uses [uv](https://docs.astral.sh/uv/) to
create the virtualenv and install dependencies.

```bash
./run.sh
# → http://127.0.0.1:8000
```

Manual setup:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # edit the VLM settings
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then drop a video onto the upload panel. The default is **1 frame per second**, a **900-frame
cap**, and batches of **4 frames per VLM request**.

### Try it without a model

Use the deterministic mock provider for an end-to-end run with no GPU or model server:

```bash
VIDFRAME_VLM_MODE=mock .venv/bin/python -m uvicorn app.main:app --port 8000
```

The mock produces plausible descriptions and is intended for UI/pipeline testing, not real analysis.

---

## Connect a vision model

vidFrame speaks the OpenAI `/v1/chat/completions` protocol with base64 image content, so any
compatible server works.

### vLLM (recommended for throughput)

```bash
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --max-model-len 32768 \
  --limit-mm-per-prompt image=8
```

```dotenv
VIDFRAME_VLM_BASE_URL=http://localhost:8000/v1
VIDFRAME_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VIDFRAME_VLM_API_KEY=EMPTY
VIDFRAME_VLM_MODE=openai
```

Use `--limit-mm-per-prompt` at least as large as your batch size. Larger models (Qwen2.5-VL-32B,
Llama 3.2 Vision, InternVL, MiniCPM-V, etc.) generally produce better on-screen text transcription.

### Ollama

```bash
ollama pull qwen2.5vl:7b

# Give the model enough context for batched frames + JSON output.
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

```dotenv
VIDFRAME_VLM_BASE_URL=http://localhost:11434/v1
VIDFRAME_VLM_MODEL=qwen2.5vl:7b
VIDFRAME_VLM_API_KEY=ollama
VIDFRAME_VLM_MODE=openai
```

Ollama's default context window is often only 4096 tokens, which truncates batched frame
descriptions and the compiled report. Set `OLLAMA_CONTEXT_LENGTH` (or the model's `num_ctx`)
generously; vidFrame also repairs truncated JSON and retries compilation, but a larger context
gives more complete reports.

### OpenRouter

Hosted vision models can be used alongside a local backend. vidFrame sends standard
`/chat/completions` requests with base64 images and adds OpenRouter's `HTTP-Referer` and
`X-Title` attribution headers.

```dotenv
# Keep the key on the server; it is never sent to the browser.
VIDFRAME_OPENROUTER_API_KEY=sk-or-v1-...
# OPENROUTER_API_KEY=sk-or-v1-...   # also accepted

VIDFRAME_OPENROUTER_MODEL=qwen/qwen2.5-vl-72b-instruct
VIDFRAME_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Choose **OpenRouter** in the upload panel or Settings to use it. The model list in Settings is
fetched from OpenRouter when a key is configured. If no key is set, the UI reports the provider as
unavailable and local jobs continue to work. The server-side key is never included in
`/api/config` or any other response.

### Speech and vector diagrams

The VLM reads diagram crops and extracts labeled geometry. The application compiles validated
geometry into SVG; it does not call an image generation model. Configure an OpenAI-compatible speech server
with `VIDFRAME_STT_BASE_URL`, `VIDFRAME_STT_API_KEY`, `VIDFRAME_STT_MODEL`, and
`VIDFRAME_STT_TIMEOUT`; it is called at `/audio/transcriptions` and auto-detects language unless a
job provides one. Source crops remain provenance artifacts when a diagram cannot be drawn reliably.

### One-key OpenRouter multimodal setup

OpenRouter also exposes `/audio/transcriptions` under the same API base and accepts
the same key used for visual analysis. Set the following in `.env`; job-level overrides remain
available, but are not required:

```dotenv
VIDFRAME_VLM_PROVIDER=openrouter
VIDFRAME_OPENROUTER_API_KEY=sk-or-v1-...
VIDFRAME_OPENROUTER_MODEL=qwen/qwen2.5-vl-72b-instruct
VIDFRAME_STT_PROVIDER=openrouter
VIDFRAME_STT_MODEL=openai/whisper-large-v3
```

With these provider flags, vidFrame derives the STT base URL and credential from
`VIDFRAME_OPENROUTER_BASE_URL` and `VIDFRAME_OPENROUTER_API_KEY`; individual
`VIDFRAME_STT_*` values still override them for another compatible service.

### Other servers

LM Studio, llama.cpp server, LocalAI, and hosted OpenAI-compatible endpoints work the same way:
set `VIDFRAME_VLM_BASE_URL`, `VIDFRAME_VLM_MODEL`, and (if required) `VIDFRAME_VLM_API_KEY`.

The UI's **Settings** dialog shows the active endpoint/model and can query the server's model list.

---

## Using the web UI

1. **Upload** a video and optionally change sample rate, frame cap, provider, vision model,
   diagram model, batch size, and timeout.
2. Watch live progress: extraction → scene detection → frame analysis → diagram detail pass →
   compilation. New frames are logged individually, and the **Live notes** panel grows as each
   batch completes.
3. Read the **Report** tab, which includes an executive overview, key moments, a visual timeline,
   a **Diagrams** section with cropped images and extracted values, recurring elements, an
   on-screen text index, and method/limitations notes.
4. Use **Timeline** to jump between scenes, **Frames** to search or filter observations (including
   diagram text and values), and the frame lightbox to inspect a still or its diagram crop.
5. The **⋯** menu offers:
   - **Recompile note** — rebuild the report from stored observations (fast, no images re-sent)
   - **Re-run vision analysis** — re-send stored frames to the VLM (and redo diagrams)
   - **Reprocess video…** — re-extract at a new sample rate from the kept source
   - **Delete analysis**
6. **Export .md** downloads the report; the menu also exports JSON, frame observations, the
   frame index, and a **diagram images zip**. All old reports remain under `versions/`.

---

## CLI

```bash
# Analyze with the configured backend
.venv/bin/python -m app.cli analyze video.mp4 --out report.md

# Offline demo
.venv/bin/python -m app.cli analyze video.mp4 --mock --fps 1 --out report.md

# Print structured JSON instead
.venv/bin/python -m app.cli analyze video.mp4 --json
```

Useful flags: `--fps`, `--max-frames`, `--batch-size`, `--provider`, `--model`, `--compile-model`,
`--timeout`, `--out`, `--json`, `--mock`.

To generate the synthetic demo video used by the tests:

```bash
.venv/bin/python scripts/make_demo_video.py --out demo.mp4
```

---

## How it works

### Pipeline

1. **Probe** — video duration, FPS, dimensions, and a sampling plan capped by `max_frames`.
2. **Extract** — seek to each sample time, resize the longest side to `max_image_side`
   (default 768px), and write JPEG frames.
3. **Deduplicate** — compare a 64-bit dHash of each frame to the last kept frame. Near-identical
   frames (Hamming distance ≤ 2) are marked as duplicates and skipped by the vision model.
4. **Segment scenes** — a larger dHash distance (≥ 14) starts a new scene. Scene boundaries drive
   the timeline and filters.
5. **Analyze** — send non-duplicate frames in batches to the selected provider (local endpoint or
   OpenRouter) with a strict JSON schema: scene, summary, description, setting, people, objects,
   actions, verbatim screen text, visual notes, confidence, and an optional diagram object.
   Requests run concurrently (`VIDFRAME_MAX_CONCURRENCY`), with tolerant JSON extraction and a
   repair retry. Each completed frame emits an SSE event and rewrites `live-notes.md`.
6. **Diagram pass** — diagrams detected in the frame pass are cropped from their frame using the
   normalized bounding box. Each crop is sent to the diagram model with surrounding frame context.
   The structured result (type, title, description, nodes, edges, labels, values, and optional
   normalized geometry) is merged back into the observation. Crops are stored under `diagrams/`;
   validated geometry is drawn as SVG under `generated-diagrams/`.
7. **Compile** — build a digest of observations, including every diagram. Short videos use a
   single pass; long ones use chunked map-reduce notes, then a final JSON report. The compiler
   validates/repairs the result, fills missing tail coverage, preserves all detected diagrams with
   their image paths, and falls back to a deterministic report if the model cannot produce valid
   JSON.
8. **Store** — write `note.md`, `note.json`, and a timestamped copy under `versions/`.

### Job modes

| Mode | Reuses | Use case |
| --- | --- | --- |
| `full` | — | Upload/reprocess: extract, analyze, compile |
| `analyze` | Stored frames | Try a new vision model on existing frames |
| `compile` | Frames + observations | Recompile/rewrite the report cheaply |

### Job artifacts

```text
data/
  vidframe.db                 # SQLite job/event store
  jobs/<job_id>/
    source.mp4                # kept for reprocessing (unless disabled)
    frames/                   # sampled JPEGs
    frames.json               # frame index, times, dHash, scene, duplicate flags
    observations.jsonl        # one structured VLM observation per analyzed frame
    live-notes.md             # incremental notes rewritten during analysis
    diagrams/                 # cropped JPEGs for every detected diagram
    diagrams.zip              # generated on first diagram export
    note.md / note.json       # latest report
    versions/<UTC stamp>.md   # timestamped report history
    versions/<UTC stamp>.json
    error.log                 # traceback for failed jobs
```

---

## API

All endpoints are under `/api`. Interactive docs are available at `/docs`.

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness and active VLM mode |
| `GET` | `/api/config` | Public defaults/limits (never includes API keys) |
| `GET` | `/api/vlm/models?provider=local\|openrouter` | Health-check and list models for a provider |
| `POST` | `/api/jobs` | Upload and start a job (multipart form) |
| `GET` | `/api/jobs` | List recent jobs |
| `GET` | `/api/jobs/{id}` | Job status, config, availability flags |
| `GET` | `/api/jobs/{id}/events` | SSE stream of log + status events |
| `POST` | `/api/jobs/{id}/cancel` | Request cancellation |
| `POST` | `/api/jobs/{id}/pause` | Pause at a safe checkpoint and retain progress |
| `POST` | `/api/jobs/{id}/resume` | Resume a paused or interrupted job from saved progress |
| `POST` | `/api/jobs/{id}/recompile` | Compile from stored observations |
| `POST` | `/api/jobs/{id}/reanalyze` | Analyze stored frames again |
| `POST` | `/api/jobs/{id}/reprocess` | Re-extract with new options |
| `DELETE` | `/api/jobs/{id}` | Delete the job and its files |
| `GET` | `/api/jobs/{id}/frames` | Frame index + observations (`scene`, `duplicates`, `q`, `limit`, `offset`) |
| `GET` | `/api/jobs/{id}/frames/{index}/image` | Frame JPEG |
| `GET` | `/api/jobs/{id}/diagrams/{name}` | Cropped diagram JPEG |
| `GET` | `/api/jobs/{id}/observations` | Merged frame/observation list |
| `GET` | `/api/jobs/{id}/timeline` | Scene groups with start/end times |
| `GET` | `/api/jobs/{id}/live-notes` | Incremental Markdown notes produced during analysis |
| `GET` | `/api/jobs/{id}/note?format=md\|json` | Latest report |
| `GET` | `/api/jobs/{id}/download?kind=note-md\|note-json\|observations\|frames\|diagrams` | File or diagram-zip download |
| `GET` | `/api/jobs/{id}/search?q=...` | Search observation text |

Example upload:

```bash
curl -F "file=@demo.mp4" \
     -F "sample_fps=1" \
     -F "batch_size=4" \
     -F "provider=openrouter" \
     -F "model=qwen/qwen2.5-vl-72b-instruct" \
     http://127.0.0.1:8000/api/jobs
```

Recompile without re-sending any images:

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/<job_id>/recompile
```

---

## Configuration

Copy `.env.example` to `.env`; environment variables override file values.

| Variable | Default | Purpose |
| --- | --- | --- |
| `VIDFRAME_VLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VIDFRAME_VLM_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Vision model |
| `VIDFRAME_VLM_API_KEY` | `EMPTY` | API key |
| `VIDFRAME_VLM_MODE` | `openai` | `openai` or `mock` |
| `VIDFRAME_VLM_PROVIDER` | `local` | Default provider: `local` or `openrouter` |
| `VIDFRAME_OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter API base |
| `VIDFRAME_OPENROUTER_API_KEY` | *(unset)* | Server-side OpenRouter key (also `OPENROUTER_API_KEY`) |
| `VIDFRAME_OPENROUTER_MODEL` | `qwen/qwen2.5-vl-72b-instruct` | Default OpenRouter vision model |
| `VIDFRAME_OPENROUTER_REFERER` | `http://127.0.0.1:8000` | `HTTP-Referer` attribution header |
| `VIDFRAME_COMPILE_MODEL` | *(vision model)* | Optional text/vision model for compilation |
| `VIDFRAME_FPS` | `1.0` | Default sample rate |
| `VIDFRAME_MAX_FRAMES` | `900` | Default frame cap |
| `VIDFRAME_BATCH_SIZE` | `4` | Frames per VLM request |
| `VIDFRAME_MAX_IMAGE_SIDE` | `768` | Resize longest image side |
| `VIDFRAME_JPEG_QUALITY` | `82` | Frame JPEG quality |
| `VIDFRAME_REQUEST_TIMEOUT` | `180` | Per-request timeout (seconds) |
| `VIDFRAME_MAX_CONCURRENCY` | `4` | Concurrent VLM requests per job |
| `VIDFRAME_MAX_CONCURRENT_JOBS` | `3` | Jobs processed simultaneously; additional jobs remain queued |
| `VIDFRAME_DATA_DIR` | `./data` | Database and artifacts |
| `VIDFRAME_MAX_UPLOAD_MB` | `4096` | Upload limit |
| `VIDFRAME_KEEP_SOURCE` | `true` | Keep source video for reprocessing |
| `VIDFRAME_COMPILE_CHUNK_CHARS` | `12000` | Map-reduce chunk size |

---

## Testing

```bash
# Full offline pipeline + API smoke test (mock VLM)
.venv/bin/python scripts/smoke_test.py

# Regenerate the demo clip
.venv/bin/python scripts/make_demo_video.py --out demo.mp4
```

The smoke test verifies extraction, duplicate skipping, diagram crop analysis, transcript storage,
and successful full, recompile, and reanalyze runs. Unit tests verify vector rendering and saved
geometry reuse.

---

## Limitations

- **Visual only.** vidFrame cannot hear audio. Dialogue, narration, and music are not analyzed, and
  the report should be combined with a transcript when speech matters.
- **Sampling misses action.** At 1 fps, brief cuts, fast gestures, and text shown for less than a
  second can be missed. Increase the sample rate for dense material.
- **OCR is model-dependent.** Small text, stylized fonts, subtitles, and low-contrast UI may be
  transcribed incorrectly. Verify important values against the frames.
- **No video understanding of motion per se.** Each frame is described independently; the compiler
  reconstructs continuity from timestamps. It may not infer complex motion between frames.
- **Context limits.** Very long videos produce large digests. Compilation uses chunking, but weak
  or small-context models can truncate output; vidFrame repairs/retries and fills gaps, at the cost
  of detail.
- **Duplicate skipping assumes visual similarity.** Slow fades or repeated slides can be merged;
  all sampled frames remain browsable in the UI.
- **Not a security boundary.** Run it on a trusted network. Uploads and generated Markdown are
  handled defensively, but no authentication is included.

---

## Project layout

```text
app/
  main.py       FastAPI routes, SSE, uploads, downloads
  pipeline.py   Extraction → analysis → compilation orchestration
  video.py      OpenCV probing, sampling, dHash dedupe, scene detection
  vlm.py        Local/OpenRouter client, batching, diagram pass, mock provider, repair
  diagrams.py   Diagram normalization, bbox crops, dedupe, digest, detail merge
  study.py      Topic and subtopic notes, visual placement, Markdown and PDF export
  vector_spec.py Validated SVG primitive renderer
  compiler.py   Legacy report compiler (unused by the active pipeline)
  prompts.py    Frame/compile/repair prompts
  db.py         SQLite jobs and events
  config.py     Environment-driven settings
  cli.py        `python -m app.cli`
scripts/
  make_demo_video.py
  smoke_test.py
static/
  index.html, app.js, styles.css, vendor/ (marked + DOMPurify)
```
