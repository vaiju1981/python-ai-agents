# Model Scorecard

The production analytics demo should choose models by live behavior, not by
generic benchmarks. `tools/model_scorecard.py` runs the same scorecard against
candidate Ollama models and records both final-answer quality and tool-call
behavior.

Run from the repository root:

```bash
PYTHONPATH=src /opt/homebrew/Caskroom/miniconda/base/envs/mlv2Py3/bin/python \
  tools/model_scorecard.py \
  --models gemma4:31b-cloud ornith:latest hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0 \
  --output-dir model_scorecards \
  --timeout 300
```

The harness writes timestamped JSON and Markdown reports under
`model_scorecards/`.

## What It Tests

- `simple`: exact instruction following.
- `complex`: small arithmetic/reasoning without tools.
- `multi-turn`: session memory across two turns.
- `single-tool`: one function call with correct arguments.
- `multi-tool`: sequential lookup + calculation tool use.
- `analytics`: choosing a metric and dimension for grouped analytics.
- `analytics-complex`: derived metric ranking.
- `complex-tool`: SQL-shaped analytics fallback through a read-only tool.

Each case scores final answer terms, completion, required tool calls, and
important tool arguments. For analytics selection, treat missing required tool
calls as blockers even when the model gives a plausible-sounding answer.

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

Latest run: `model_scorecards/ollama_model_scorecard_20260707T045650Z.md`

| Model | Score | Duration | Read |
| --- | ---: | ---: | --- |
| `ornith:latest` | `800/800` | `88.5s` | Best production-demo candidate. Perfect across simple, memory, tool, multi-tool, analytics, and SQL-style cases. |
| `gemma4:31b-cloud` | `765/800` | `8.9s` | Strong fast fallback. Excellent tool arguments and clean output, but missed one numeric detail after a complex SQL-style tool. |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | `510/800` | `46.5s` | Useful for tool-call stress testing. It often called tools correctly, but leaked `<think>` traces and failed to synthesize final answers after tools. |

In the latest run, `ornith:latest` was `9.98x` slower than
`gemma4:31b-cloud` (`88.5s` vs `8.9s`). The previous captured run showed a
narrower spread (`66.9s` vs `21.1s`, about `3.18x`), so treat latency as
workload and service-state dependent. Accuracy still favors `ornith:latest`
for the flagship demo.

## Recommendation

Use `ornith:latest` as the default model for the production analytics demo.
Use `gemma4:31b-cloud` as the speed/fallback candidate, especially when the
demo flow uses structured analytics tools rather than free-form SQL. Keep
`hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` as a regression target for
tool-call handling, not as the primary demo model.
