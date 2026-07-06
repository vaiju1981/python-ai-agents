# Ollama Model Scorecard

| Model | Score | Pass Rate | Duration | Status |
| --- | ---: | ---: | ---: | --- |
| `gemma4:31b-cloud` | 74.0/80.0 (92.5%) | 87.5% | 10.3s | ok |
| `ornith:latest` | 80.0/80.0 (100.0%) | 100.0% | 94.1s | ok |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | 52.0/80.0 (65.0%) | 50.0% | 40.1s | ok |

## Case Details

### gemma4:31b-cloud

| Case | Category | Score | Tools | Detail |
| --- | --- | ---: | --- | --- |
| simple_exact | simple | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| complex_reasoning | complex | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| multi_turn_memory | multi-turn | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| single_tool_addition | single-tool | 10.0/10.0 PASS | add_numbers | answer terms ok; completed; required tools ok; arguments: numeric arguments ok |
| multi_tool_metric_delta | multi-tool | 10.0/10.0 PASS | lookup_metric, lookup_metric, subtract_numbers | answer terms ok; completed; required tools ok; arguments: north_lookup=True south_lookup=True subtract_120_90=True |
| analytics_groupby | analytics | 10.0/10.0 PASS | run_metric_query | answer terms ok; completed; required tools ok; arguments: argument pairs ok |
| analytics_margin_rank | analytics-complex | 10.0/10.0 PASS | rank_metric | answer terms ok; completed; required tools ok; arguments: argument pairs ok |
| complex_sql_tool | complex-tool | 4.0/10.0 FAIL | read_only_sql | missing answer terms: 95; completed; required tools ok; arguments: sql terms ok |

### ornith:latest

| Case | Category | Score | Tools | Detail |
| --- | --- | ---: | --- | --- |
| simple_exact | simple | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| complex_reasoning | complex | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| multi_turn_memory | multi-turn | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| single_tool_addition | single-tool | 10.0/10.0 PASS | add_numbers | answer terms ok; completed; required tools ok; arguments: numeric arguments ok |
| multi_tool_metric_delta | multi-tool | 10.0/10.0 PASS | lookup_metric, lookup_metric, subtract_numbers | answer terms ok; completed; required tools ok; arguments: north_lookup=True south_lookup=True subtract_120_90=True |
| analytics_groupby | analytics | 10.0/10.0 PASS | run_metric_query | answer terms ok; completed; required tools ok; arguments: argument pairs ok |
| analytics_margin_rank | analytics-complex | 10.0/10.0 PASS | rank_metric | answer terms ok; completed; required tools ok; arguments: argument pairs ok |
| complex_sql_tool | complex-tool | 10.0/10.0 PASS | read_only_sql | answer terms ok; completed; required tools ok; arguments: sql terms ok |

### hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0

| Case | Category | Score | Tools | Detail |
| --- | --- | ---: | --- | --- |
| simple_exact | simple | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| complex_reasoning | complex | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| multi_turn_memory | multi-turn | 10.0/10.0 PASS | - | answer terms ok; completed; no tool required |
| single_tool_addition | single-tool | 4.0/10.0 FAIL | add_numbers | missing answer terms: 42; completed; required tools ok; arguments: numeric arguments ok |
| multi_tool_metric_delta | multi-tool | 4.0/10.0 FAIL | lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, lookup_metric, subtract_numbers | missing answer terms: 30; completed; required tools ok; arguments: north_lookup=True south_lookup=True subtract_120_90=True |
| analytics_groupby | analytics | 10.0/10.0 PASS | run_metric_query, run_metric_query | answer terms ok; completed; required tools ok; arguments: argument pairs ok |
| analytics_margin_rank | analytics-complex | 3.0/10.0 FAIL | rank_metric | missing answer terms: gadget; completed; required tools ok; arguments: missing argument pairs: rank_metric.dimension=product |
| complex_sql_tool | complex-tool | 1.0/10.0 FAIL | - | missing answer terms: gadget, 95; completed; missing tools: read_only_sql; arguments: no SQL argument |

## Selection Heuristic

- Prefer the highest score for the production analytics demo, but inspect analytics, multi-tool, and complex-tool rows first; those are more predictive than simple echo tests.
- Use the fastest passing model for local demos when analytics-tool accuracy is tied.
- Treat failures to call required tools as blockers for the analytics demo, even if the final answer looks plausible.