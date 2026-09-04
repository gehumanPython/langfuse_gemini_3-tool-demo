# System Design — Langfuse + Gemini AI Research Assistant

## 1. Overview

This project is a **proof-of-concept for LLM observability** in an enterprise setting. It demonstrates how an organisation can instrument AI pipelines with full traceability — capturing every prompt, response, token count, latency, and cost — without changing business logic.

**Core use case:** A user inputs any topic; a 3-step AI agent pipeline (Research → Critique → Report) produces a structured executive brief. Every step is fully observable in a Langfuse dashboard.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER'S BROWSER                           │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │              Streamlit Web UI  (app.py)                 │   │
│   │                                                         │   │
│   │  [Text Input: Topic]  →  [▶ Run Agent Pipeline]        │   │
│   │                                                         │   │
│   │  Step 1 — Research     (streaming text output)          │   │
│   │  Step 2 — Critique     (streaming text output)          │   │
│   │  Step 3 — Report       (streaming text output)          │   │
│   │                                                         │   │
│   │  [Metrics Panel: tokens | latency | cost | trace link]  │   │
│   └──────────────────────┬──────────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────────┘
                           │ HTTP (Streamlit server)
                           │
          ┌────────────────▼────────────────┐
          │         Python Backend           │
          │   (runs on Streamlit server)     │
          │                                  │
          │  ┌──────────────────────────┐    │
          │  │   Agent Pipeline Logic   │    │
          │  │                          │    │
          │  │  stream_gemini()         │    │
          │  │  ├─ 1-research span      │    │
          │  │  ├─ 2-critique span      │    │
          │  │  └─ 3-report span        │    │
          │  └────────┬─────────┬───────┘    │
          └───────────┼─────────┼────────────┘
                      │         │
           ┌──────────▼──┐  ┌───▼──────────────┐
           │ Google Gemini│  │  Langfuse Cloud   │
           │ API          │  │                   │
           │              │  │  Traces           │
           │ gemini-3.6   │  │  Spans            │
           │ -flash       │  │  Token counts     │
           │ (streaming)  │  │  Latency          │
           └─────────────-┘  │  Cost estimates   │
                             └───────────────────┘
```

---

## 3. Component Breakdown

### 3.1 `app.py` — Streamlit Web Application

The primary interface for the executive demo. Runs as a web server and serves a browser-based UI.

| Responsibility | Implementation |
|---|---|
| Secret management | `get_secret()` — reads from `st.secrets` (cloud) or `.env` (local) |
| LLM streaming | `stream_gemini()` — generator yielding text chunks |
| Retry logic | `tenacity` — separate policies for 503 (fast) and 429 (slow) |
| Langfuse tracing | `start_as_current_observation()` context manager per step |
| UI rendering | `st.write_stream()` for streaming, `st.metric()` for KPIs |

### 3.2 `main.py` — CLI Pipeline

A command-line version of the same pipeline using the OpenAI-compatible Gemini endpoint. Demonstrates the `@observe` decorator pattern for automatic tracing. Useful for scripted or scheduled runs.

| Responsibility | Implementation |
|---|---|
| LLM calls | `langfuse.openai.OpenAI` — auto-instrumented drop-in wrapper |
| Step tracing | `@observe()` decorator on each pipeline function |
| Root trace | `@observe(name="mock-agent-pipeline")` on `run_agent()` |
| OTel metadata | `LangfuseOtelSpanAttributes` for user ID, session ID, tags |

### 3.3 Langfuse (Observability Layer)

Langfuse v4 uses **OpenTelemetry (OTel)** as its tracing standard. When `Langfuse()` is initialized, it registers an OTel exporter that ships spans to Langfuse Cloud asynchronously.

Each pipeline run creates a **trace** (top-level) containing 3 **spans** (one per step):

```
Trace: executive-demo-pipeline
├── Span: 1-research    (generation)
│     input:  { system, prompt }
│     output: { summary text }
│     usage:  { input_tokens, output_tokens }
│     latency: Xs
│
├── Span: 2-critique    (generation)
│     input:  { system, prompt (includes summary) }
│     output: { critique text }
│     usage:  { input_tokens, output_tokens }
│     latency: Xs
│
└── Span: 3-report      (generation)
      input:  { system, prompt (includes summary + critique) }
      output: { report text }
      usage:  { input_tokens, output_tokens }
      latency: Xs
```

### 3.4 Google Gemini API

| Property | Value |
|---|---|
| Model | `gemini-3.6-flash` |
| API endpoint (`app.py`) | Native `google-genai` SDK (streaming) |
| API endpoint (`main.py`) | OpenAI-compatible: `generativelanguage.googleapis.com/v1beta/openai/` |
| Free tier limits | 15 RPM · 1,500 RPD |
| Streaming | Yes — chunks yielded progressively via `generate_content_stream()` |

---

## 4. Data Flow

```
User enters topic
       │
       ▼
app.py creates root Langfuse span ("executive-demo-pipeline")
       │
       ├──► Step 1: Research
       │         └─ stream_gemini() opens child span "1-research"
       │         └─ Calls Gemini API (streaming)
       │         └─ Chunks yielded → st.write_stream() renders in browser
       │         └─ Final chunk carries token usage metadata
       │         └─ Span closed → data exported to Langfuse Cloud (async)
       │
       ├──► Step 2: Critique  (receives Step 1 output as context)
       │         └─ Same flow as Step 1, span "2-critique"
       │
       └──► Step 3: Report    (receives Step 1 + Step 2 output as context)
                 └─ Same flow, span "3-report"
                 └─ Root span closed, langfuse.flush() called
                 └─ Metrics panel rendered
                 └─ Langfuse trace URL displayed to user
```

---

## 5. Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| UI | Streamlit | 1.63+ | Browser-based interactive web app |
| LLM | Google Gemini 3.6 Flash | — | Language model for generation |
| LLM SDK | google-genai | 2.22+ | Native Gemini SDK with streaming |
| LLM SDK (CLI) | openai | 3.7+ | OpenAI-compatible Gemini endpoint |
| Observability | Langfuse | 4.15+ | Tracing, token tracking, cost |
| Tracing standard | OpenTelemetry | 1.44+ | OTel spans/exporters (Langfuse v4) |
| Retry logic | tenacity | 8.2+ | Exponential backoff for API errors |
| Config | python-dotenv | 1.2+ | Load `.env` for local development |
| Language | Python | 3.12 | — |
| Deployment | Streamlit Community Cloud | — | Free public hosting from GitHub |

---

## 6. Observability Model

Langfuse captures the following for every pipeline run:

| Signal | What is captured | Where visible |
|---|---|---|
| **Trace** | Full pipeline run, topic input, final report output | Traces list |
| **Spans** | Each of the 3 steps with individual timing | Trace detail view |
| **Prompts** | Exact system + user prompt sent to Gemini | Span detail |
| **Completions** | Exact text returned by Gemini | Span detail |
| **Token usage** | Input tokens, output tokens per step | Span + aggregated |
| **Latency** | Time-to-first-token + total per step | Timeline view |
| **Cost** | Estimated USD cost per step and total | Dashboard |
| **Tags** | `["demo", "executive", "streamlit"]` | Filterable |
| **User ID** | `"executive-demo"` | Filterable |

This enables the organisation to answer questions like:
- Which pipeline step is slowest?
- What is the average cost per report?
- Which prompts produce the best quality outputs?
- How does quality change over time?

---

## 7. Error Handling Strategy

| Error | HTTP Code | Cause | Retry Strategy |
|---|---|---|---|
| Server overload | 503 | Gemini high demand | 4 retries, 3–30s exponential backoff |
| Rate limit | 429 | Free tier quota exceeded | 3 retries, 60–120s exponential backoff |
| Auth failure | 400/401 | Invalid API key | No retry — shown as inline error |
| Fallback | Any | All retries exhausted | Friendly inline message, no crash |

---

## 8. Secret Management

| Environment | How secrets are stored | How they are read |
|---|---|---|
| Local development | `.env` file (git-ignored) | `python-dotenv` → `os.getenv()` |
| Streamlit Community Cloud | Streamlit Secrets (encrypted) | `st.secrets[key]` |
| Future: enterprise | Environment variables / Vault | Same `get_secret()` helper |

The `get_secret()` helper in `app.py` tries `st.secrets` first, then falls back to `os.getenv()`, making the same code work in both environments without modification.

---

## 9. Deployment

### Local
```
venv/Scripts/streamlit run app.py
```
Runs at `http://localhost:8501`. Reads secrets from `.env`.

### Streamlit Community Cloud (current)
- GitHub repo: `gehumanPython/langfuse_gemini_3-tool-demo`
- Branch: `main`, File: `app.py`
- Secrets injected via Streamlit Cloud dashboard
- Auto-redeploys on every `git push` to `main`

### Future Enterprise Options
| Option | Trade-offs |
|---|---|
| Azure App Service / AWS ECS | Full control, scalable, costs money |
| Docker container | Portable, self-hosted, full isolation |
| Kubernetes | High-availability, complex setup |

---

## 10. Known Limitations & Future Improvements

| Limitation | Impact | Suggested Fix |
|---|---|---|
| Free Gemini tier: 15 RPM | Demo can't be run rapidly back-to-back | Enable Google Cloud billing |
| ~10–30s latency per step | UI feels slow despite streaming | Use Gemini's paid tier or enterprise endpoint |
| No authentication on Streamlit app | Anyone with the URL can use it | Add Streamlit's built-in auth or SSO |
| No conversation history | Each run is independent | Add `st.session_state` to persist context |
| Single fixed pipeline | Only 3 hardcoded steps | Make steps configurable via UI |
| No prompt versioning | Prompt changes are untracked | Use Langfuse Prompt Management |
| No automated evaluation | Quality is subjective | Add Langfuse LLM-as-a-judge evaluators |
| Cost displayed locally only | No historical cost tracking | Query Langfuse API for aggregated cost reports |

---

## 11. Repository Structure

```
langfuse-mock-project/
│
├── app.py                # Streamlit web application (primary demo)
├── main.py               # CLI pipeline (alternative entry point)
├── requirements.txt      # Direct dependencies only (for deployment)
├── .gitignore            # Excludes .env and venv/
├── SYSTEM_DESIGN.md      # This document
│
├── .env                  # (git-ignored) Local secrets
└── venv/                 # (git-ignored) Python virtual environment
```

---

*Document generated: September 2026 | Project: LLM Observability PoC | Owner: AI Infrastructure*
