# Model Scorecard

The production analytics demo should choose models by live behavior, not by
generic benchmarks. `tools/model_scorecard.py` runs the same scorecard against
candidate Ollama models through the actual `demos.analytics` stack: CSV import,
profiling, semantic model creation, the demo agent prompt, `AnalyticsToolset`,
and `ModelsToolset`.

Run from the repository root:

```bash
PYTHONPATH=src /opt/homebrew/Caskroom/miniconda/base/envs/mlv2Py3/bin/python \
  tools/model_scorecard.py \
  --output-dir model_scorecards \
  --timeout 60 \
  --num-ctx 65536
```

The harness writes timestamped JSON and Markdown reports under
`model_scorecards/`.

## What It Tests

- `schema_discovery`: dataset/table/metric discovery through `describe_dataset`.
- `revenue_by_region`: semantic grouped query through `run_query`.
- `profit_by_product_sql`: read-only SQL fallback through `run_sql`.
- `revenue_trend`: time-series analysis through `trend`.
- `summarize_profit`: descriptive statistics through `summarize`.
- `build_predictive_model`: predictive modeling through `build_model`.
- `forecast_revenue`: forecasting through `forecast`.
- `campaign_ab_test`: experiment comparison through `ab_test`.
- `causal_effect`: observational causal estimate through `causal_effect`.
- `cluster_segments`: segmentation through `cluster`.

Each case scores final answer terms, completion, required tool calls, and
important tool arguments. The argument checks expect discovered
`table.column` references from the demo semantic model, not hardcoded generic
field names. Treat missing required tool calls as blockers even when the model
gives a plausible-sounding answer.

## Detailed Rubric

Each case is scored out of 100:

| Component | Points | Meaning |
| --- | ---: | --- |
| Answer correctness | 35 | Expected facts/terms appear in the final answer. |
| Completion | 10 | Agent turn completed without hitting an error/step limit. |
| Tool selection | 20 | Required analytics demo tools were called. |
| Tool arguments | 20 | Tool arguments used discovered table/column refs. |
| Tool efficiency | 5 | No excessive repeated tool calls. |
| Output hygiene | 10 | Final answer is clean: no visible `<think>` trace and not empty. |

Tool-call correctness and final-answer quality are scored separately. A model
gets substantial credit for calling the right tool with the right arguments,
but the case should also verify that the user-facing answer includes the
important result or caveat. The causal case checks the substance of the caveat
(`causation` plus `confound`) rather than requiring the literal word `caveat`.

## Current Run

Latest run: `model_scorecards/ollama_model_scorecard_20260707T055223Z.md`

| Model | Score | Duration | Read |
| --- | ---: | ---: | --- |
| `gemma4:31b-cloud` | `1000/1000` | `25.4s` | Best current production-demo candidate when cloud access is acceptable: perfect score and lowest latency. |
| `ornith:latest` | `1000/1000` | `144.6s` | Strong local/offline candidate: perfect score with correct tool routing, arguments, and final answers. |

This run caps Ollama context with `num_ctx=65536` to avoid local memory pressure.
With that cap, `ornith:latest` was `5.71x` slower than `gemma4:31b-cloud`
(`144.6s` vs `25.4s`). Accuracy is tied on the current analytics scorecard;
latency favors `gemma4:31b-cloud`, while `ornith:latest` remains the best
local/offline candidate.

## Recommendation

Use `gemma4:31b-cloud` as the cloud-tier default and `ornith:latest` as the
local/offline default. Compare within the tier you can actually deploy; only look
across tiers for reference.
