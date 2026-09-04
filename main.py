"""
Langfuse + Gemini Mock Project
-------------------------------
Demonstrates LLM observability with Langfuse Cloud using:
  - Gemini (via OpenAI-compatible endpoint, free tier)
  - langfuse.openai drop-in wrapper (auto-traces every LLM call)
  - @observe decorator (traces multi-step agent logic as spans)

Langfuse dashboard will show: traces, spans, token usage, latency, cost.

Langfuse SDK version: 4.x (OpenTelemetry-based)
"""

import os
from dotenv import load_dotenv, find_dotenv
from langfuse import Langfuse, observe, get_client, LangfuseOtelSpanAttributes
from langfuse.openai import OpenAI          # drop-in for openai.OpenAI
from opentelemetry import trace as otel_trace

# ── Load environment variables from .env (searches current dir + parents) ────
load_dotenv(find_dotenv(usecwd=True) or find_dotenv())

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
# Support both LANGFUSE_HOST and LANGFUSE_BASE_URL (Langfuse dashboard shows the latter)
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

# ── Initialise Langfuse (sets up OTel exporter to Langfuse Cloud) ─────────────
langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

# ── Gemini via OpenAI-compatible endpoint ─────────────────────────────────────
# langfuse.openai wraps this client and auto-sends every call to your dashboard
client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = "gemini-3.6-flash"   # free-tier Gemini model


# ─────────────────────────────────────────────────────────────────────────────
# Helper – single LLM call (automatically traced by langfuse.openai wrapper)
# ─────────────────────────────────────────────────────────────────────────────
def ask_gemini(prompt: str, system: str = "You are a helpful assistant.") -> str:
    """Single LLM call — Langfuse records prompt, completion, tokens & latency."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Multi-step agent pipeline (each step is a nested span via @observe)
# ─────────────────────────────────────────────────────────────────────────────
@observe()   # creates a "research_topic" span inside the parent trace
def research_topic(topic: str) -> str:
    """Simulate a research sub-step: ask Gemini to summarise a topic."""
    lf = get_client()
    lf.update_current_span(name="research-step", input={"topic": topic})

    result = ask_gemini(
        prompt=f"In 2 sentences, summarise the key points about: {topic}",
        system="You are a concise research assistant.",
    )

    lf.update_current_span(output={"summary": result})
    return result


@observe()   # creates a "critique_summary" span
def critique_summary(summary: str) -> str:
    """Simulate a critique sub-step: ask Gemini to find gaps."""
    result = ask_gemini(
        prompt=f"List one potential gap or limitation in this summary:\n\n{summary}",
        system="You are a critical reviewer.",
    )
    return result


@observe()   # creates a "generate_report" span
def generate_report(topic: str, summary: str, critique: str) -> str:
    """Simulate a final synthesis step."""
    result = ask_gemini(
        prompt=(
            f"Write a 3-sentence report on '{topic}'.\n"
            f"Use this summary: {summary}\n"
            f"And address this critique: {critique}"
        ),
        system="You are a professional report writer.",
    )
    return result


@observe(name="mock-agent-pipeline")   # top-level trace / root span
def run_agent(topic: str) -> dict:
    """
    Three-step agent pipeline:
      1. Research   → summarise the topic
      2. Critique   → identify a gap
      3. Report     → synthesise into a final answer
    All steps appear as nested spans in the Langfuse trace.
    """
    print(f"\n[Agent] Starting pipeline for topic: '{topic}'")

    # Attach trace-level metadata via OTel span attributes (Langfuse v4 style)
    current_span = otel_trace.get_current_span()
    current_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_USER_ID, "demo-user")
    current_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_SESSION_ID, "mock-session-001")
    current_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_TAGS, ["demo", "gemini", "mock-agent"])
    current_span.set_attribute(LangfuseOtelSpanAttributes.TRACE_METADATA, f'{{"topic": "{topic}"}}')

    summary  = research_topic(topic)
    critique = critique_summary(summary)
    report   = generate_report(topic, summary, critique)

    return {
        "topic":    topic,
        "summary":  summary,
        "critique": critique,
        "report":   report,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── Demo 1: Simple single call ────────────────────────────────────────────
    print("=" * 60)
    print("Demo 1 — Simple LLM call (auto-traced by Langfuse)")
    print("=" * 60)
    answer = ask_gemini("What is the capital of France? Answer in one word.")
    print(f"Answer: {answer}")

    # ── Demo 2: Multi-step agent pipeline ────────────────────────────────────
    print("\n" + "=" * 60)
    print("Demo 2 — Multi-step agent pipeline (nested spans in Langfuse)")
    print("=" * 60)
    result = run_agent("Langfuse and LLM observability")

    print(f"\n[Summary]  {result['summary']}")
    print(f"[Critique] {result['critique']}")
    print(f"[Report]   {result['report']}")

    # ── Flush all traces to Langfuse Cloud before exit ────────────────────────
    langfuse.flush()
    print("\nTraces sent to Langfuse Cloud. Check your dashboard at:")
    print("  https://cloud.langfuse.com")
