# Confidence-Based Tool Routing — Eval Harness

Custom evaluation harness for the "Confidence-Based Tool Routing in Agentic
AI" project. Phase 1 built the measurement infrastructure before the
confidence module and tool router existed, so every later experiment logs
into the same schema from day one. Phase 2 added the four confidence
estimators; Phase 3 adds the router that acts on them.

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
    confidence.py Confidence estimators: token entropy, self-consistency,
                  external verifier, hybrid combiner
    router.py     The routing decision + threshold sweep / selection
  examples/
    dummy_run.py  Simulates 240 synthetic tasks end-to-end, no API calls
  tests/
    test_metrics.py     hand-verified unit tests for every metric
    test_costs.py       pricing table + staleness guard
    test_confidence.py  all four estimators
    test_router.py      routing decision, sweep, threshold selection
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

- **The router owns the cut, not the tool choice.** `route()` answers
  DIRECT or TOOL from a confidence and a threshold; which tool a TOOL
  decision calls is the task's category, which the caller knows and the
  router does not. Missing confidence escalates by default — no evidence
  of confidence is not evidence of confidence.
- **Threshold sweeps report routing quality, not CPST.** Precision,
  recall and the call rates depend only on the necessity label and the
  decision, so they can be recomputed counterfactually from one run's
  log. Correctness under a decision the system never took cannot be, so
  a real CPST curve needs the pipeline run at each threshold.

## Next (Phase 4)

Everything above is pure: estimators take an already-completed API
response, the router takes an already-computed confidence. What is still
missing is the runner that makes the actual calls — query in, TaskRecord
out — wiring `confidence.py` and `router.py` to a real model and real
tools. After that, the cascade: entropy on every task, escalate to
self-consistency or the verifier only near the threshold, which is where
the cost argument of the project actually gets made.
