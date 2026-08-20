# Confidence-Based Tool Routing — Eval Harness (Phase 1)

Custom evaluation harness for the "Confidence-Based Tool Routing in Agentic
AI" project. This is Phase 1 of the research execution roadmap (Fig. 2):
build the measurement infrastructure before the confidence module and tool
router exist, so every later experiment logs into the same schema from day
one.

## Structure

```
confidence_routing/
  eval_harness/
    models.py    TaskRecord — the one schema everything logs into
    costs.py      OpenAI pricing table + cost estimation
    logger.py     Append-only JSONL run logger
    metrics.py    CPST, ECE (+ grouping), routing precision/recall,
                  unnecessary/missed-call rate, latency breakdown
    report.py     Aggregates records -> summary dict / printable report
  examples/
    dummy_run.py  Simulates 240 synthetic tasks end-to-end, no API calls
  tests/
    test_metrics.py  11 hand-verified unit tests for every metric
```

## Quickstart

```bash
pip install -r requirements.txt   # currently empty — stdlib only so far
python -m examples.dummy_run      # simulate + report, no API key needed
python -m pytest tests/ -v        # verify the math
```

## Design notes

- **TaskRecord is the only contract.** Every future piece (confidence
  module, tool router, tool connectors) just needs to produce one
  TaskRecord per task. Nothing downstream cares how the record was
  produced — metrics.py and report.py are pure functions over the schema.
- **JSONL, not a database.** One line per task means a crashed run is
  still readable up to the last completed line, and logs are diffable /
  greppable / concatenable with plain shell tools. Move to sqlite/parquet
  later if volume demands it — the RunLogger interface won't need to
  change for callers.
- **CPST is USD-based by default.** `(tokens + API cost) / correct tasks`
  from the roadmap is interpreted as "total dollar cost (token cost +
  any flat tool-API cost) divided by correct tasks" — dollars are the
  actionable unit. Pure token-count CPST is reported alongside for
  comparability with token-only metrics in related work (e.g.
  DataGovBench's Tokens-per-Successful-Task).
- **ECE supports arbitrary grouping.** Pass `group_by=lambda r: ...` to
  `compute_ece` to slice by tool type, task category, or anything else in
  `TaskRecord.meta` — this is what the cross-tool calibration-transfer
  experiment (Fig. 2, Experiment A) will use: calibrate on one tool
  category, evaluate ECE on another.
- **Pricing table needs periodic verification.** OpenAI revises rates.
  `costs.py` documents this at the top — check
  https://openai.com/api/pricing before trusting CPST numbers in a
  results table.

## Next (Phase 2)

Wire in the confidence module (token entropy, self-consistency, external
lightweight verifier) so `TaskRecord.confidence_score` is populated by a
real signal instead of `dummy_run.py`'s simulated one. The report/metrics
code needs no changes — it already expects exactly this field.
