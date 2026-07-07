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
  --models gemma4:31b-cloud ornith:latest hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0 \
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
| Tool selection | 20 | Required tools were called, or no unexpected tools were called. |
| Tool arguments | 20 | Tool arguments matched the requested metric, dimension, SQL terms, or numeric values. |
| Tool efficiency | 5 | No excessive repeated tool calls. |
| Output hygiene | 10 | Final answer is clean: no visible `<think>` trace and not empty. |

The output hygiene row is intentionally strict. Some models, especially
`hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0`, may return valid tool calls
while leaving the user-visible response inside `<think>...</think>` or failing
to produce a final answer after tools complete. That is scored separately from
tool-call correctness because it is a product-demo failure even when the tool
arguments are correct.

## Current Run

Latest run: `model_scorecards/ollama_model_scorecard_20260707T053638Z.md`

| Model | Score | Duration | Read |
| --- | ---: | ---: | --- |
| `gemma4:31b-cloud` | `1000/1000` | `33.1s` | Best current production-demo candidate for this analytics harness: perfect tool routing, arguments, and final answers. |
| `ornith:latest` | `930/1000` | `149.0s` | Strong local candidate. It routed tools correctly but missed final-answer terms on modeling and causal caveat cases. |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | `640/1000` | `175.3s` | Useful stress/regression target. It still leaks thinking traces, repeats tool calls, and misses some required tools. |

This run caps Ollama context with `num_ctx=65536` to avoid local memory pressure.
With that cap, `ornith:latest` was `4.50x` slower than `gemma4:31b-cloud`
(`149.0s` vs `33.1s`). Accuracy and latency both favor `gemma4:31b-cloud` for
the current production analytics scorecard, while `ornith:latest` remains a
good local/offline candidate.

## Recommendation

Use `gemma4:31b-cloud` as the default production-demo model when cloud access is
acceptable. Use `ornith:latest` as the local/offline candidate. Keep
`hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` as a regression target for
tool-call handling, not as the primary demo model.
