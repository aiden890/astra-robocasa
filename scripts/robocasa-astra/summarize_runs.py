"""Summarise the real Astra runs listed in runs/astra_ledger.jsonl as a Markdown table (no model calls).

    python3 scripts/robocasa-astra/summarize_runs.py > context_test_docs/results_table.md
"""

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "runs" / "astra_ledger.jsonl"
WEEKLY_CREDITS_ESTIMATE = (50_000, 60_000)  # community estimate for Pro; not an official figure


def main() -> None:
    rows, totals = [], {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "credits": 0.0, "usd": 0.0}
    for line in LEDGER.read_text().splitlines():
        entry = json.loads(line)
        run = Path(entry["run_dir"])
        summary_path = run / "summary.json"
        if not summary_path.exists():
            rows.append(f"| {entry['task']} | {entry['shots']} | (no summary yet: running or interrupted) | | | | | | `{run.name}` |")
            continue
        s = json.loads(summary_path.read_text())
        u, c = s["usage"], s["cost"]
        for key in ("input_tokens", "cached_input_tokens", "output_tokens"):
            totals[key] += u.get(key, 0)
        totals["credits"] += c["credits"]
        totals["usd"] += c["usd_api_equivalent"]
        rows.append(
            f"| {s['task']} | {s['shots']} | {'success' if s['success'] else 'fail'} ({s['finished_reason']}) | "
            f"{s['turns']} | {s['steps_used']}/{s['step_budget']} | {u.get('input_tokens', 0):,} "
            f"({c['cache_hit_ratio']:.0%} cached) | {u.get('output_tokens', 0):,} | {c['credits']:.1f} / ${c['usd_api_equivalent']:.2f} | "
            f"`runs/robodawn/{run.name}` |"
        )
    print("| task | shots | result | turns | steps | input tokens | output tokens | credits / API $ | run dir |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    lo, hi = WEEKLY_CREDITS_ESTIMATE
    print(f"\nTotal: input {totals['input_tokens']:,} (cached {totals['cached_input_tokens']:,}), output "
          f"{totals['output_tokens']:,}; {totals['credits']:.1f} credits = ${totals['usd']:.2f} API-equivalent; "
          f"{totals['credits'] / hi:.1%}-{totals['credits'] / lo:.1%} of an estimated Pro weekly allowance "
          f"({lo:,}-{hi:,} credits, community estimate).")


if __name__ == "__main__":
    main()
