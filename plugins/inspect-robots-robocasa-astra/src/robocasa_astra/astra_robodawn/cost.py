"""Token usage -> subscription credits and API-equivalent dollars for gpt-6-astra.

Rates (per 1M tokens), checked 2026-10-06:
* credits: OpenAI ChatGPT rate card (Work and Codex): input 250, cached input 25, output 1,250;
  Codex does not charge cache writes.
* dollars: OpenAI API standard pricing: input $10, cached input $1, output $50.

``input_tokens`` from Codex includes the cached part. Reasoning tokens are assumed to be part of
``output_tokens`` (as in the Responses API); ``*_if_reasoning_separate`` gives the upper bound in
case they are not.
"""

from __future__ import annotations

CREDITS_PER_MILLION = {"input": 250.0, "cached_input": 25.0, "output": 1250.0}
USD_PER_MILLION = {"input": 10.0, "cached_input": 1.0, "output": 50.0}
USAGE_KEYS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")


def add_usage(total: dict, usage: dict) -> dict:
    """Accumulate Codex ``turn.completed`` usage dicts (missing keys count as 0)."""
    for key in USAGE_KEYS:
        total[key] = total.get(key, 0) + int(usage.get(key) or 0)
    return total


def _price(usage: dict, rates: dict, extra_output: int = 0) -> float:
    cached = int(usage.get("cached_input_tokens") or 0)
    uncached = max(int(usage.get("input_tokens") or 0) - cached, 0)
    output = int(usage.get("output_tokens") or 0) + extra_output
    return (uncached * rates["input"] + cached * rates["cached_input"] + output * rates["output"]) / 1_000_000


def estimate(usage: dict) -> dict:
    """Credits and API-equivalent dollars for one usage dict (rounded for reports)."""
    reasoning = int(usage.get("reasoning_output_tokens") or 0)
    return {
        "credits": round(_price(usage, CREDITS_PER_MILLION), 2),
        "usd_api_equivalent": round(_price(usage, USD_PER_MILLION), 4),
        "credits_if_reasoning_separate": round(_price(usage, CREDITS_PER_MILLION, reasoning), 2),
        "usd_if_reasoning_separate": round(_price(usage, USD_PER_MILLION, reasoning), 4),
        "cache_hit_ratio": round((usage.get("cached_input_tokens") or 0) / max(usage.get("input_tokens") or 0, 1), 3),
    }
