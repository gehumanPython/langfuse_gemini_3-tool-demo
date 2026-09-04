"""
Langfuse + Gemini — Executive Demo (Streaming)
-----------------------------------------------
Streamlit UI with:
  - Streaming LLM responses (text appears word-by-word)
  - Live elapsed-time counter per step
  - Observability metrics panel (tokens, latency, cost)
  - Direct link to the Langfuse trace
"""

import time
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

import streamlit as st
from google import genai as google_genai
from google.genai import errors as genai_errors
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langfuse import Langfuse

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
)

# ── Secret loader: Streamlit Cloud (st.secrets) → local .env fallback ─────────
def get_secret(key: str, default: str = "") -> str:
    """Works both on Streamlit Community Cloud and locally."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

# ── Client setup ──────────────────────────────────────────────────────────────
gemini = google_genai.Client(api_key=get_secret("GEMINI_API_KEY"))

langfuse_client = Langfuse(
    public_key=get_secret("LANGFUSE_PUBLIC_KEY"),
    secret_key=get_secret("LANGFUSE_SECRET_KEY"),
    host=get_secret("LANGFUSE_HOST") or get_secret("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
)

MODEL = "gemini-3.6-flash"

# ── Gemini pricing (per 1M tokens) ───────────────────────────────────────────
INPUT_COST_PER_M  = 0.075
OUTPUT_COST_PER_M = 0.30


# ── Detect error type for user-friendly messages ─────────────────────────────
def _is_rate_limit(exc: Exception) -> bool:
    return isinstance(exc, genai_errors.ClientError) and "429" in str(exc)

def _is_server_error(exc: Exception) -> bool:
    return isinstance(exc, genai_errors.ServerError)


# ── Streaming call — yields text chunks, logs to Langfuse manually ────────────
def stream_gemini(system: str, prompt: str, step_name: str, metrics: dict):
    """
    Streams Gemini response into the UI and records the span in Langfuse.
    Returns the full response text when done.
    """
    full_text    = ""
    input_tokens = 0
    out_tokens   = 0

    with langfuse_client.start_as_current_observation(
        name=step_name,
        as_type="generation",
        input={"system": system, "prompt": prompt},
        model=MODEL,
    ) as span:
        t0 = time.time()

        # 503 server errors: retry quickly (3–30s backoff)
        @retry(
            retry=retry_if_exception_type(genai_errors.ServerError),
            wait=wait_exponential(multiplier=2, min=3, max=30),
            stop=stop_after_attempt(4),
            reraise=True,
        )
        # 429 rate limit: retry slowly (60–120s backoff — free tier resets per minute)
        @retry(
            retry=retry_if_exception_type(genai_errors.ClientError),
            wait=wait_exponential(multiplier=2, min=60, max=120),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call_with_retry():
            return list(gemini.models.generate_content_stream(
                model=MODEL,
                contents=prompt,
                config={"system_instruction": system},
            ))

        try:
            chunks = _call_with_retry()
        except genai_errors.ClientError as e:
            if "429" in str(e):
                yield (
                    "**Rate limit hit (429).** The free Gemini API allows ~15 requests/min. "
                    "Please wait 1–2 minutes and try again."
                )
            else:
                yield f"**API error:** {e}"
            return
        except genai_errors.ServerError as e:
            yield f"**Gemini server unavailable after retries.** Please try again shortly. ({e})"
            return

        for chunk in chunks:
            if chunk.text:
                full_text += chunk.text
                yield chunk.text

            # Capture token counts from the final chunk
            if chunk.usage_metadata:
                if chunk.usage_metadata.prompt_token_count:
                    input_tokens = chunk.usage_metadata.prompt_token_count
                if chunk.usage_metadata.candidates_token_count:
                    out_tokens = chunk.usage_metadata.candidates_token_count

        elapsed = round(time.time() - t0, 2)

        # Update the Langfuse span with output + usage
        span.update(
            output={"text": full_text},
            usage_details={"input": input_tokens, "output": out_tokens},
        )

    # Accumulate into shared metrics dict
    metrics["input_tokens"]  += input_tokens
    metrics["output_tokens"] += out_tokens
    metrics.setdefault("step_times", {})[step_name] = elapsed


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.title("AI Research Assistant")
st.caption("Powered by Gemini · Observability by Langfuse")
st.info(
    "**Free tier limits:** Gemini allows ~15 requests/min and 1,500/day. "
    "Each pipeline run uses 3 requests. Wait **1–2 minutes** between runs to avoid rate limits.",
    icon="ℹ️",
)
st.divider()

topic   = st.text_input(
    "Enter a topic or question:",
    placeholder="e.g. Impact of AI on supply chain management",
)
run_btn = st.button("▶  Run Agent Pipeline", type="primary", disabled=not bool(topic))
st.caption("Each step may take 10–30s. If Gemini is busy (503), it retries automatically up to 4 times.")
st.divider()

if run_btn and topic:
    metrics   = {"input_tokens": 0, "output_tokens": 0, "step_times": {}}
    t_start   = time.time()
    trace_url = None

    # Wrap everything in a single Langfuse trace
    with langfuse_client.start_as_current_observation(
        name="executive-demo-pipeline",
        as_type="agent",
        input={"topic": topic},
    ) as root_span:

        from langfuse import get_client
        lf       = get_client()
        trace_id = lf.get_current_trace_id()
        if trace_id:
            trace_url = lf.get_trace_url(trace_id=trace_id)

        # ── Step 1: Research ─────────────────────────────────────────────────
        st.subheader("Step 1 — Research")
        with st.spinner("Researching... (streaming)"):
            summary = st.write_stream(stream_gemini(
                system="You are a concise research assistant.",
                prompt=f"In 3 sentences, summarise the key points about: {topic}",
                step_name="1-research",
                metrics=metrics,
            ))
        t1 = metrics["step_times"].get("1-research", 0)
        st.caption(f"Completed in {t1}s · {metrics['input_tokens']} input tokens so far")

        # ── Step 2: Critique ─────────────────────────────────────────────────
        st.subheader("Step 2 — Critical Analysis")
        with st.spinner("Critiquing... (streaming)"):
            critique = st.write_stream(stream_gemini(
                system="You are a critical analyst.",
                prompt=f"In 2 sentences, identify the most important gap in this summary:\n\n{summary}",
                step_name="2-critique",
                metrics=metrics,
            ))
        t2 = metrics["step_times"].get("2-critique", 0)
        st.caption(f"Completed in {t2}s")

        # ── Step 3: Executive Report ──────────────────────────────────────────
        st.subheader("Step 3 — Executive Report")
        with st.spinner("Writing report... (streaming)"):
            report = st.write_stream(stream_gemini(
                system="You are a senior business analyst writing for executives.",
                prompt=(
                    f"Write a concise 4-sentence executive report on '{topic}'.\n"
                    f"Build on this summary: {summary}\n"
                    f"Address this critique: {critique}\n"
                    f"End with a clear recommendation."
                ),
                step_name="3-report",
                metrics=metrics,
            ))
        t3 = metrics["step_times"].get("3-report", 0)
        st.caption(f"Completed in {t3}s")

        root_span.update(output={"report": report})

    langfuse_client.flush()
    total_time = round(time.time() - t_start, 2)

    # ── Metrics panel ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Observability Metrics")

    out_tok   = metrics["output_tokens"]
    in_tok    = metrics["input_tokens"]
    total_tok = in_tok + out_tok
    cost      = (in_tok / 1_000_000) * INPUT_COST_PER_M + (out_tok / 1_000_000) * OUTPUT_COST_PER_M

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Input Tokens",  f"{in_tok:,}")
    col2.metric("Output Tokens", f"{out_tok:,}")
    col3.metric("Total Tokens",  f"{total_tok:,}")
    col4.metric("Total Time",    f"{total_time}s")
    col5.metric("Est. Cost",     f"${cost:.5f}")

    st.divider()
    if trace_url:
        st.link_button("Open Full Trace in Langfuse →", trace_url, type="primary")
        st.caption("See every prompt, completion, token count and latency breakdown for each step.")
    else:
        st.link_button("Open Langfuse Dashboard →", "https://cloud.langfuse.com", type="primary")
