# Ollama Analytics Demo Model Scorecard

Harness: `demos.analytics` CSV source, profiler, semantic model, demo agent, `AnalyticsToolset`, and `ModelsToolset`.

| Model | Score | Pass Rate | Duration | Status |
| --- | ---: | ---: | ---: | --- |
| `gemma4:31b-cloud` | 1000.0/1000.0 (100.0%) | 100.0% | 33.1s | ok |
| `ornith:latest` | 930.0/1000.0 (93.0%) | 80.0% | 149.0s | ok |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | 640.0/1000.0 (64.0%) | 40.0% | 175.3s | ok |

## Speed Comparison

| Model | Duration | Relative To Fastest |
| --- | ---: | ---: |
| `gemma4:31b-cloud` | 33.1s | 1.00x |
| `ornith:latest` | 149.0s | 4.50x |
| `hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0` | 175.3s | 5.29x |

## Rubric

| Component | Points | Meaning |
| --- | ---: | --- |
| Answer correctness | 35 | Expected facts/terms appear in the final answer. |
| Completion | 10 | Agent turn completed without hitting an error/step limit. |
| Tool selection | 20 | Required analytics demo tools were called. |
| Tool arguments | 20 | Tool arguments used discovered table/column refs. |
| Tool efficiency | 5 | No excessive repeated tool calls. |
| Output hygiene | 10 | Final answer is clean: no visible `<think>` trace and not empty. |

## Case Details

### gemma4:31b-cloud

| Case | Category | Score | Tools | Warnings | Detail |
| --- | --- | ---: | --- | --- | --- |
| schema_discovery | analytics-schema | 100.0/100.0 PASS | describe_dataset | - | answer terms ok; completed; required tools ok; efficiency: tool calls efficient; hygiene: clean final answer |
| revenue_by_region | analytics-query | 100.0/100.0 PASS | run_query | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| profit_by_product_sql | analytics-sql | 100.0/100.0 PASS | run_sql | - | answer terms ok; completed; required tools ok; arguments: sql terms ok; efficiency: tool calls efficient; hygiene: clean final answer |
| revenue_trend | analytics-time | 100.0/100.0 PASS | trend | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| summarize_profit | analytics-stats | 100.0/100.0 PASS | summarize | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| build_predictive_model | analytics-modeling | 100.0/100.0 PASS | build_model | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| forecast_revenue | analytics-forecast | 100.0/100.0 PASS | forecast | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| campaign_ab_test | analytics-experiment | 100.0/100.0 PASS | ab_test | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| causal_effect | analytics-causal | 100.0/100.0 PASS | causal_effect | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| cluster_segments | analytics-segmentation | 100.0/100.0 PASS | cluster | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |

### ornith:latest

| Case | Category | Score | Tools | Warnings | Detail |
| --- | --- | ---: | --- | --- | --- |
| schema_discovery | analytics-schema | 100.0/100.0 PASS | describe_dataset | - | answer terms ok; completed; required tools ok; efficiency: tool calls efficient; hygiene: clean final answer |
| revenue_by_region | analytics-query | 100.0/100.0 PASS | run_query | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| profit_by_product_sql | analytics-sql | 100.0/100.0 PASS | run_sql | - | answer terms ok; completed; required tools ok; arguments: sql terms ok; efficiency: tool calls efficient; hygiene: clean final answer |
| revenue_trend | analytics-time | 100.0/100.0 PASS | trend | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| summarize_profit | analytics-stats | 100.0/100.0 PASS | summarize | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| build_predictive_model | analytics-modeling | 65.0/100.0 FAIL | build_model | - | missing answer terms: feature; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| forecast_revenue | analytics-forecast | 100.0/100.0 PASS | forecast | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| campaign_ab_test | analytics-experiment | 100.0/100.0 PASS | ab_test | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| causal_effect | analytics-causal | 65.0/100.0 FAIL | causal_effect | - | missing answer terms: caveat; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| cluster_segments | analytics-segmentation | 100.0/100.0 PASS | cluster | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |

### hf.co/RefinedNeuro/RefinedToolCallV5-3b:Q8_0

| Case | Category | Score | Tools | Warnings | Detail |
| --- | --- | ---: | --- | --- | --- |
| schema_discovery | analytics-schema | 50.0/100.0 FAIL | describe_dataset, run_query, run_query | - | missing answer terms: scorecard_sales, revenue, profit; stop_reason=model_error; required tools ok; efficiency: too many calls (3 > 2); hygiene: clean final answer |
| revenue_by_region | analytics-query | 65.0/100.0 FAIL | run_query | - | missing answer terms: east, 1218; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| profit_by_product_sql | analytics-sql | 90.0/100.0 PASS | run_sql | thinking trace leaked without final answer | answer terms ok; completed; required tools ok; arguments: sql terms ok; efficiency: tool calls efficient; hygiene: thinking trace leaked without final answer |
| revenue_trend | analytics-time | 90.0/100.0 PASS | trend | thinking trace leaked before final answer | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: thinking trace leaked before final answer |
| summarize_profit | analytics-stats | 90.0/100.0 PASS | summarize | thinking trace leaked before final answer | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: thinking trace leaked before final answer |
| build_predictive_model | analytics-modeling | 50.0/100.0 FAIL | build_model, build_model, build_model | - | missing answer terms: feature, importance; stop_reason=model_error; required tools ok; arguments: argument refs ok; efficiency: too many calls (3 > 2); hygiene: clean final answer |
| forecast_revenue | analytics-forecast | 90.0/100.0 PASS | forecast, forecast | thinking trace leaked without final answer | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: thinking trace leaked without final answer |
| campaign_ab_test | analytics-experiment | 10.0/100.0 FAIL | run_query, run_query, run_query, run_query | - | missing answer terms: difference; stop_reason=max_steps; missing tools: ab_test; arguments: missing refs: ab_test.metric contains scorecard_sales.revenue, ab_test.groupColumn contains scorecard_sales.campaign, ab_test.groupA contains test, ab_test.groupB contains control; efficiency: too many calls (4 > 2); hygiene: clean final answer |
| causal_effect | analytics-causal | 50.0/100.0 FAIL | causal_effect, causal_effect, causal_effect | thinking trace leaked without final answer | missing answer terms: causation; completed; required tools ok; arguments: argument refs ok; efficiency: too many calls (3 > 2); hygiene: thinking trace leaked without final answer |
| cluster_segments | analytics-segmentation | 55.0/100.0 FAIL | cluster | thinking trace leaked without final answer | missing answer terms: cluster, silhouette; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: thinking trace leaked without final answer |

## Selection Heuristic

- Prefer the highest score for the production analytics demo, but inspect modeling, forecast, SQL, and causal rows first.
- Use the fastest passing model when analytics-tool accuracy is tied.
- Treat failures to call required demo tools as blockers, even if the final answer sounds plausible.