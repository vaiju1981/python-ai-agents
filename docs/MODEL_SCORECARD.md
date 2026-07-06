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

## Current Run

Latest run: `model_scorecards/ollama_model_scorecard_20260706T231448Z.md`

| Model | Score | Read |
| --- | ---: | --- |
| `ornith:latest` | `80/80` | Best production-demo candidate. Perfect across simple, memory, tool, multi-tool, and analytics cases. Slower than the other two. |
| `gemma4:31b-cloud` | `74/80` | Strong general fallback. Fastest in this run and excellent at tool arguments, but missed one numeric detail after a complex SQL-style tool. |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | `52/80` | Useful for narrow tool-call experiments, but not ready as the flagship analytics-demo model: it often called tools yet failed to synthesize the final answer. |

## Recommendation

Use `ornith:latest` as the default model for the production analytics demo.
Use `gemma4:31b-cloud` as the speed/fallback candidate, especially when the
demo flow uses structured analytics tools rather than free-form SQL. Keep
`hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` as a regression target for
tool-call handling, not as the primary demo model.
