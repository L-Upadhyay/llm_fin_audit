# Live evaluation — ollama:llama3.2

| Condition | Answers | Checkable claims / answer | Wrong claims | Answers with ≥1 wrong claim | Answers with no checkable numbers | Unsourced claims / answer | Median latency |
|---|---|---|---|---|---|---|---|
| LLM only (no data) | 93 | 2.22 | 97.1% (94%–99%) | 78.5% (69%–86%) | 21.5% | 0.12 | 6.5s |
| Agent team (tools, unverified) | 93 | 1.02 | 50.5% (41%–60%) | 14.0% (8%–22%) | 72.0% | 0.03 | 11.0s |
| Pipeline — first attempt | 93 | 4.26 | 3.3% (2%–6%) | 12.9% (8%–21%) | 1.1% | 0 | 4.0s |
| Pipeline — after 1 rewrite | 93 | 4.19 | 1.0% (0%–3%) | 4.3% (2%–11%) | 1.1% | 0 | 4.0s |
| Pipeline — delivered (after correction, re-verified) | 93 | 4.19 | 0.0% (0%–1%) | 0.0% (0%–4%) | 1.1% | 0 | 4.0s |
