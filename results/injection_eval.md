# Injected-error evaluation of the claim verifier

248 synthetic narratives over the facts snapshot (2026-09-26T05:56:13+00:00), seed 664. 676 injected errors among 1831 numeric/verdict statements.

| Metric | Value |
|---|---|
| Precision | 100.0% |
| Recall (all) | 88.9% |
| False-positive rate on clean statements | 0.0% |
| Spurious contradictions (no matching statement) | 0 |

**Recall by template family** (standard = phrasings the extractor targets; hard = phrasings it was not designed for)

| Family | Injected | Recall | Clean coverage |
|---|---|---|---|
| hard | 75 | 0.0% | 0.0% |
| standard | 601 | 100.0% | 100.0% |

**Recall by error type**

| Error | Injected | Recall |
|---|---|---|
| comparison_flip | 49 | 100.0% |
| metric_swap | 123 | 84.5% |
| near_miss | 102 | 89.2% |
| scale | 108 | 87.0% |
| sign_flip | 96 | 85.4% |
| ticker_swap | 106 | 84.9% |
| unit_confusion | 38 | 97.4% |
| verdict_flip | 54 | 100.0% |

**Recall by template**

| Template | Injected | Recall |
|---|---|---|
| comparison | 49 | 100.0% |
| heading_bullet | 52 | 100.0% |
| long_distance | 18 | 0.0% |
| paraphrase | 32 | 0.0% |
| prose_is | 55 | 100.0% |
| prose_of | 89 | 100.0% |
| table | 183 | 100.0% |
| two_ticker | 119 | 100.0% |
| value_first | 25 | 0.0% |
| verdict | 54 | 100.0% |
