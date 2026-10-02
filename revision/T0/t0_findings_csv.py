"""Write revision/out/T0/findings.csv from the findings register in the Task 0 README.

Each finding gets the Task 0b step (or later task) that is expected to address it.
Run after editing the README:  python revision/T0/t0_findings_csv.py
"""
import csv
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "revision" / "out" / "T0"
AREA = {"D": "data", "P": "preprocessing/models", "S": "statistics", "E": "economics",
        "R": "figures/reproducibility"}
ACTION = {
    "D1": "0b.1 correct documented sources", "D2": "0b.1 fix META mapping (decision 2)",
    "D3": "0b.1 split-adjust price features", "D4": "0b.1 fix BKM formula",
    "D5": "0b.1 fix feat_50 contract key", "D6": "0b.1 GOOG earnings (WRDS)",
    "D7": "0b.1 enforce or drop the 50-contract claim (decision 4)", "D8": "decision 3 (sample window); paper wording",
    "D9": "paper wording", "D10": "paper/README wording", "D11": "0b.2 drop legacy split join",
    "D12": "0b.1/0b.5 build RV before any join", "D13": "0b.1 current-GICS sector map", "D14": "0b.8 run commands", "D15": "0b R9 remove future-event features",
    "P1": "0b.2 per-window preprocessing", "P2": "0b.2 surface NaN treatment (decision 6)",
    "P3": "0b.3 OLS in float64", "P4": "0b.4 constant-forecast rule (decision 5)",
    "P5": "0b.6 flat book on ties (decision 5)", "P6": "0b.3 document or fit on full block",
    "P7": "0b.2 purge the early-stopping fold", "P8": "0b.5 neural baselines not rerun; 0b.6 family",
    "P9": "0b.6 trial family; paper wording", "S1": "0b.4 canonical daily cross-sectional IC",
    "S2": "0b.4 NaN rule", "S3": "Task 1/2 corrected SE", "S4": "Task 4 cross-sectional naming",
    "S5": "0b.7 moving-block bootstrap", "S6": "paper wording", "S7": "Task 1 rerun on corrected outputs",
    "S8": "paper wording (paired lift SE)", "S9": "paper wording", "S10": "Task 5 calibration",
    "S11": "paper wording", "E1": "0b.6 arithmetic Sharpe", "E2": "0b.6 PBO family",
    "E3": "0b.6 ETF-consistent book", "E4": "0b.6 SR0 family; DSR wording", "E5": "Task 3 sensitivity",
    "E6": "0b.8 number-to-script map", "R1": "0b.8 figure paths", "R2": "0b.8 figures read outputs",
    "R3": "0b.5 rerun TreeSHAP on benchmark models", "R4": "0b.3/0b.8 reproducibility check",
    "R5": "0b.8 .gitignore", "R6": "0b.8 run commands", "R7": "0b.3 version pins",
    "R8": "0b.8 remove or guard rm -rf", "R9": "0b.8 provenance", "R10": "0b.8 number-to-script map",
}
row_re = re.compile(r"^\| ([DPSER]\d+) \| ([BMWI]) \| (.+) \| (.+) \|\s*$")
rows = []
for line in (OUT / "README.md").read_text().splitlines():
    m = row_re.match(line)
    if m:
        fid, sev, text, evid = m.groups()
        rows.append({"id": fid, "area": AREA[fid[0]], "severity": sev,
                     "finding": re.sub(r"\*\*|`", "", text).replace("<br>", " "),
                     "evidence": re.sub(r"`", "", evid), "addressed_by": ACTION.get(fid, "")})
with open(OUT / "findings.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} findings written; by severity:",
      {s: sum(r['severity'] == s for r in rows) for s in 'BMWI'})
