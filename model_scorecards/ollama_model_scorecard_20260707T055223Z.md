# Ollama Analytics Demo Model Scorecard

Harness: `demos.analytics` CSV source, profiler, semantic model, demo agent, `AnalyticsToolset`, and `ModelsToolset`.

| Model | Score | Pass Rate | Duration | Status |
| --- | ---: | ---: | ---: | --- |
| `gemma4:31b-cloud` | 1000.0/1000.0 (100.0%) | 100.0% | 25.4s | ok |
| `ornith:latest` | 1000.0/1000.0 (100.0%) | 100.0% | 144.6s | ok |

## Speed Comparison

| Model | Duration | Relative To Fastest |
| --- | ---: | ---: |
| `gemma4:31b-cloud` | 25.4s | 1.00x |
| `ornith:latest` | 144.6s | 5.71x |

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
| build_predictive_model | analytics-modeling | 100.0/100.0 PASS | build_model | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| forecast_revenue | analytics-forecast | 100.0/100.0 PASS | forecast | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| campaign_ab_test | analytics-experiment | 100.0/100.0 PASS | ab_test | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| causal_effect | analytics-causal | 100.0/100.0 PASS | causal_effect | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |
| cluster_segments | analytics-segmentation | 100.0/100.0 PASS | cluster | - | answer terms ok; completed; required tools ok; arguments: argument refs ok; efficiency: tool calls efficient; hygiene: clean final answer |

## Selection Heuristic

- Prefer the highest score for the production analytics demo, but inspect modeling, forecast, SQL, and causal rows first.
- Use the fastest passing model when analytics-tool accuracy is tied.
- Treat failures to call required demo tools as blockers, even if the final answer sounds plausible.