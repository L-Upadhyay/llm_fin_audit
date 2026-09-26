"""
Freeze live facts for a fixed ticker universe into a JSON snapshot.

Evaluation runs read the snapshot instead of yfinance, so results are
reproducible and the ground truth doesn't drift between conditions.

    python -m src.evaluation.snapshot            # writes data/fixtures/facts_snapshot.json
"""

import json
import os
import sys
from datetime import datetime, timezone

from src.data.facts import build_facts


SNAPSHOT_PATH = "data/fixtures/facts_snapshot.json"

# Mix of sectors, sizes, and financial health: mega-cap tech, autos,
# staples, retail, energy, pharma, telecom, travel, and loss-making or
# distressed names. Banks are left out: their balance sheets have no
# current ratio, so the classical layer returns INSUFFICIENT_DATA by design.
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "F", "GM", "KO", "PEP", "WMT", "TGT", "COST", "SBUX", "NKE",
    "XOM", "CVX", "JNJ", "PFE", "MRK", "INTC", "BA", "DIS",
    "T", "VZ", "CCL", "RIVN", "LCID", "PTON", "AMC",
]


def take_snapshot(tickers=UNIVERSE, path=SNAPSHOT_PATH):
    facts = {}
    for t in tickers:
        try:
            facts[t] = build_facts(t)
        except Exception as e:  # keep going; one bad ticker shouldn't sink the run
            print(f"skip {t}: {e}", file=sys.stderr)
    payload = {
        "taken_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "yfinance via src.data.facts.build_facts",
        "facts": facts,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, default=str)
    return payload


def load_snapshot(path=SNAPSHOT_PATH):
    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    snap = take_snapshot()
    verdicts = {t: f["csp_verdict"] for t, f in snap["facts"].items()}
    print(f"Wrote {SNAPSHOT_PATH} with {len(verdicts)} tickers")
    for t, v in verdicts.items():
        print(f"  {t:6} {v}")
