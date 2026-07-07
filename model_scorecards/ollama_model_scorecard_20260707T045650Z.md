# Ollama Model Scorecard

| Model | Score | Pass Rate | Duration | Status |
| --- | ---: | ---: | ---: | --- |
| `gemma4:31b-cloud` | 765.0/800.0 (95.6%) | 87.5% | 8.9s | ok |
| `ornith:latest` | 800.0/800.0 (100.0%) | 100.0% | 88.5s | ok |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | 510.0/800.0 (63.7%) | 50.0% | 46.5s | ok |

## Speed Comparison

| Model | Duration | Relative To Fastest |
| --- | ---: | ---: |
| `gemma4:31b-cloud` | 8.9s | 1.00x |
| `ornith:latest` | 88.5s | 9.98x |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | 46.5s | 5.25x |

## Rubric

| Component | Points | Meaning |
| --- | ---: | --- |
| Answer correctness | 35 | Expected facts/terms appear in the final answer. |
| Completion | 10 | Agent turn completed without hitting an error/step limit. |
| Tool selection | 20 | Required tools were called, or no unexpected tools were called. |
| Tool arguments | 20 | Tool arguments matched the requested metric, dimension, SQL terms, or numeric values. |
| Tool efficiency | 5 | No excessive repeated tool calls. |
| Output hygiene | 10 | Final answer is clean: no visible `<think>` trace and not empty. |

## Case Details

### gemma4:31b-cloud

| Case | Category | Score | Tools | Warnings | Detail |
| --- | --- | ---: | --- | --- | --- |
| simple_exact | simple | 100.0/100.0 PASS | - | - | answer terms ok; completed; no tool required; hygiene: clean final answer |
| complex_reasoning | complex | 100.0/100.0 PASS | - | - | answer terms ok; completed; no tool required; hygiene: clean final answer |
| multi_turn_memory | multi-turn | 100.0/100.0 PASS | - | - | answer terms ok; completed; no tool required; hygiene: clean final answer |
| single_tool_addition | single-tool | 100.0/100.0 PASS | add_numbers | - | answer terms ok; completed; required tools ok; arguments: numeric arguments ok; efficiency: tool calls efficient; hygiene: clean final answer |
| multi_tool_metric_delta | multi-tool | 100.0/100.0 PASS | lookup_metric, lookup_metric, subtract_numbers | - | answer terms ok; completed; required tools ok; arguments: north_lookup=True south_lookup=True subtract_120_90=True; efficiency: tool calls efficient; hygiene: clean final answer |
| analytics_groupby | analytics | 100.0/100.0 PASS | run_metric_query | - | answer terms ok; completed; required tools ok; arguments: argument pairs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| analytics_margin_rank | analytics-complex | 100.0/100.0 PASS | rank_metric | - | answer terms ok; completed; required tools ok; arguments: argument pairs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| complex_sql_tool | complex-tool | 65.0/100.0 FAIL | read_only_sql | - | missing answer terms: 95; completed; required tools ok; arguments: sql terms ok; efficiency: tool calls efficient; hygiene: clean final answer |

### ornith:latest

| Case | Category | Score | Tools | Warnings | Detail |
| --- | --- | ---: | --- | --- | --- |
| simple_exact | simple | 100.0/100.0 PASS | - | - | answer terms ok; completed; no tool required; hygiene: clean final answer |
| complex_reasoning | complex | 100.0/100.0 PASS | - | - | answer terms ok; completed; no tool required; hygiene: clean final answer |
| multi_turn_memory | multi-turn | 100.0/100.0 PASS | - | - | answer terms ok; completed; no tool required; hygiene: clean final answer |
| single_tool_addition | single-tool | 100.0/100.0 PASS | add_numbers | - | answer terms ok; completed; required tools ok; arguments: numeric arguments ok; efficiency: tool calls efficient; hygiene: clean final answer |
| multi_tool_metric_delta | multi-tool | 100.0/100.0 PASS | lookup_metric, lookup_metric, subtract_numbers | - | answer terms ok; completed; required tools ok; arguments: north_lookup=True south_lookup=True subtract_120_90=True; efficiency: tool calls efficient; hygiene: clean final answer |
| analytics_groupby | analytics | 100.0/100.0 PASS | run_metric_query | - | answer terms ok; completed; required tools ok; arguments: argument pairs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| analytics_margin_rank | analytics-complex | 100.0/100.0 PASS | rank_metric | - | answer terms ok; completed; required tools ok; arguments: argument pairs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| complex_sql_tool | complex-tool | 100.0/100.0 PASS | read_only_sql | - | answer terms ok; completed; required tools ok; arguments: sql terms ok; efficiency: tool calls efficient; hygiene: clean final answer |

### hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0

| Case | Category | Score | Tools | Warnings | Detail |
| --- | --- | ---: | --- | --- | --- |
| simple_exact | simple | 90.0/100.0 PASS | - | thinking trace leaked before final answer | answer terms ok; completed; no tool required; hygiene: thinking trace leaked before final answer |
| complex_reasoning | complex | 90.0/100.0 PASS | - | thinking trace leaked before final answer | answer terms ok; completed; no tool required; hygiene: thinking trace leaked before final answer |
| multi_turn_memory | multi-turn | 90.0/100.0 PASS | - | thinking trace leaked before final answer | answer terms ok; completed; no tool required; hygiene: thinking trace leaked before final answer |
| single_tool_addition | single-tool | 55.0/100.0 FAIL | add_numbers | thinking trace leaked without final answer | missing answer terms: 42; completed; required tools ok; arguments: numeric arguments ok; efficiency: tool calls efficient; hygiene: thinking trace leaked without final answer |
| multi_tool_metric_delta | multi-tool | 50.0/100.0 FAIL | lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, subtract_numbers | thinking trace leaked without final answer | missing answer terms: 30; completed; required tools ok; arguments: north_lookup=True south_lookup=True subtract_120_90=True; efficiency: too many calls (9 > 4); hygiene: thinking trace leaked without final answer |
| analytics_groupby | analytics | 90.0/100.0 PASS | run_metric_query, run_metric_query | thinking trace leaked before final answer | answer terms ok; completed; required tools ok; arguments: argument pairs ok; efficiency: tool calls efficient; hygiene: thinking trace leaked before final answer |
| analytics_margin_rank | analytics-complex | 35.0/100.0 FAIL | rank_metric | thinking trace leaked without final answer | missing answer terms: gadget; completed; required tools ok; arguments: missing argument pairs: rank_metric.dimension=product; efficiency: tool calls efficient; hygiene: thinking trace leaked without final answer |
| complex_sql_tool | complex-tool | 10.0/100.0 FAIL | - | thinking trace leaked without final answer | missing answer terms: gadget, 95; completed; missing tools: read_only_sql; arguments: no SQL argument; efficiency: no tool calls; hygiene: thinking trace leaked without final answer |

## Selection Heuristic

- Prefer the highest score for the production analytics demo, but inspect analytics, multi-tool, and complex-tool rows first; those are more predictive than simple echo tests.
- Use the fastest passing model for local demos when analytics-tool accuracy is tied.
- Treat failures to call required tools as blockers for the analytics demo, even if the final answer looks plausible.