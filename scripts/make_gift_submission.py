"""Build a GIFT-Eval leaderboard submission folder from a harness result directory.

    python scripts/make_gift_submission.py <eval dir> --name TimeSSM-2.5M --out submissions
    python ~/archive/gift-eval-upstream/scripts/validate_results.py submissions/TimeSSM-2.5M

The harness writes all_results.csv in the leaderboard's column order; this script fills
`domain` and `num_variates` from the leaderboard's dataset_configurations.csv, blanks MSIS
(it needs the 95 % interval, which nine quantiles from 0.1 to 0.9 do not give; the
validator accepts a blank), sets the model name, and writes config.json. The result is the
folder to add under results/<name> in a pull request to SalesforceAIResearch/gift-eval.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
CONFIGS = HERE / "configs" / "gift_dataset_configurations.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir", help="directory holding all_results.csv")
    ap.add_argument("--name", required=True, help="model name: folder name and the model column")
    ap.add_argument("--out", default="submissions")
    ap.add_argument("--model-type", default="zero-shot")
    ap.add_argument("--model-dtype", default="float32")
    ap.add_argument("--model-link", default="https://huggingface.co/IUseAMouse/timessm-2.5m")
    ap.add_argument("--code-link", default="https://github.com/IUseAMouse/TimeSSM")
    ap.add_argument("--org", default="Yvann Vincent")
    args = ap.parse_args()

    meta = {r["dataset"]: r for r in csv.DictReader(CONFIGS.open())}
    src = Path(args.eval_dir) / "all_results.csv"
    rows = list(csv.DictReader(src.open()))
    if len(rows) != len(meta):
        sys.exit(f"{src}: {len(rows)} rows, {len(meta)} expected")
    header = list(rows[0].keys())
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "all_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            r = dict(r)
            m = meta[r["dataset"]]
            r["model"] = args.name
            r["domain"] = m["domain"]
            r["num_variates"] = m["num_variates"]
            if r["eval_metrics/MSIS"].lower() in ("nan", "inf", "-inf"):
                r["eval_metrics/MSIS"] = ""
            w.writerow(r)
    (out / "config.json").write_text(json.dumps({
        "model": args.name,
        "model_type": args.model_type,
        "model_dtype": args.model_dtype,
        "model_link": args.model_link,
        "code_link": args.code_link,
        "org": args.org,
        "testdata_leakage": "No",
        "replication_code_available": "Yes",
    }, indent=4) + "\n")
    print(f"{out}: {len(rows)} rows, config.json written")


if __name__ == "__main__":
    main()
