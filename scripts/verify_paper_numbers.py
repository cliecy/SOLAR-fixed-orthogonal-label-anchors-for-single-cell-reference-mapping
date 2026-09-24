#!/usr/bin/env python3
"""Check every number printed in the SOLAR manuscript against the per-run data.

    python scripts/verify_paper_numbers.py            # summary + exit status
    python scripts/verify_paper_numbers.py --report verification_report.csv

Inputs are the ledgers in ``paper/`` (one row per printed number, with the
printed string and a machine-readable specification) and the per-run records
in ``data/``. Each value is recomputed without intermediate rounding and then
formatted to the printed precision. Static claims about hyperparameters and
protocol constants are checked against ``configs/`` and the source code.

The run passes when the set of mismatching claims equals the set documented in
``paper/known_discrepancies.csv`` (normally empty); every other mismatch, and
every documented discrepancy that no longer occurs, fails the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import paper_numbers as pn  # noqa: E402
import paper_static_checks as static  # noqa: E402

LEDGERS = ("claims_main.csv", "claims_si_text.csv", "claims_si_tables.csv", "claims_figures.csv")


def fmt(value: float, printed: str) -> str:
    """Format `value` like `printed` (decimals, explicit plus sign, NA/dash)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return printed if printed in ("NA", "–", "NA (SD –)") else "NA"
    text = printed.lstrip("+-")
    decimals = len(text.split(".")[1]) if "." in text else 0
    out = f"{value:.{decimals}f}"
    if out.startswith("-") and float(out) == 0:
        out = out[1:]
    if printed.startswith("+") and not out.startswith("-"):
        out = "+" + out
    if "," in printed:
        out = f"{int(round(value)):,}"
    return out


def expected_for(row: dict) -> str:
    spec = json.loads(row["spec"])
    kind, printed = row["kind"], row["printed"]
    stat = spec.get("stat")
    if kind == "batch":
        table = pn.batch_table(spec["method"], spec["dim"], spec["metric"], spec["population"])
        hit = table[(table.dataset_id == spec["dataset"]) & (table.heldout_batch == spec["batch"])].iloc[0]
        if stat == "n":
            return str(int(hit.n_valid))
        if stat == "seed_sd":
            return "–" if hit.n_expected == 1 else fmt(hit.seed_sd, printed)
        return fmt(hit["mean"], printed)
    if kind == "summary":
        s = pn.summary(spec["method"], spec["dim"], spec["metric"], spec["population"])
        if stat == "mean":
            return fmt(s.mean, printed)
        if stat == "sd":
            return "–" if s.batches_valid < 2 or math.isnan(s.sd) else fmt(s.sd, printed)
        if stat == "D":
            return f"{s.datasets_valid}/{s.datasets_applicable}"
        if stat == "B":
            return f"{s.batches_valid}/{s.batches_applicable}"
        if stat == "F":
            return f"{s.fits_valid}/{s.fits_expected}"
        if stat == "na_code":
            return ",".join(s.codes) if s.codes else "–"
    if kind == "contrast":
        c = pn.contrast(tuple(spec["left"]), tuple(spec["right"]), spec["metric"], spec["population"])
        values = {"L": c.left, "R": c.right, "delta": c.delta,
                  "D": f"{c.datasets_valid}/{c.datasets_applicable}",
                  "B": f"{c.batches_valid}/{c.batches_applicable}",
                  "FL": f"{c.fits_left}/{c.fits_left_expected}",
                  "FR": f"{c.fits_right}/{c.fits_right_expected}"}
        value = values[stat]
        return value if isinstance(value, str) else fmt(value, printed)
    if kind == "dataset":
        mean, valid, applicable = pn.dataset_level(spec["method"], spec["dim"], spec["dataset"],
                                                   spec["metric"], spec["population"])
        if stat == "coverage":
            return f"{valid}/{applicable}"
        if stat == "na_code":
            codes = set()
            for pop in ("reference", "query", "joint"):
                table = pn.batch_table(spec["method"], spec["dim"], spec["metric"], pop)
                table = table[table.dataset_id == spec["dataset"]]
                codes |= set(pn.na_codes(spec["metric"], pop, table)) - {"S"}
            return ",".join(sorted(codes)) if codes else "–"
        return fmt(mean, printed)
    if kind == "composition":
        frame = pn.composition()
        hit = frame[(frame.dataset_id == spec["dataset"]) & (frame.heldout_batch == spec["batch"])].iloc[0]
        return fmt(float(hit[spec["field"]]), printed)
    if kind == "width":
        return fmt(pn.width_mean(spec["dataset"], spec["dim"], spec["metric"]), printed)
    if kind == "width_dim":
        frame = pn.within_batch_width()
        rows = frame[(frame.dataset_id == spec["dataset"]) & (frame.dimension == spec["dim"])]
        dims = set(rows.actual_dimension.astype(int))
        return str(spec["dim"]) if len(rows) == len(pn.SEEDS) and dims == {spec["dim"]} else f"observed {len(rows)} runs, dims {dims}"
    if kind == "confusion":
        return fmt(float(pn.villani_confusion().loc[spec["true"], spec["pred"]]), printed)
    if kind in ("static", "assertion", "derived"):
        return static.evaluate(kind, spec, printed)
    if kind == "non_result":
        return printed
    raise ValueError(f"unhandled claim kind {kind!r} in {row['claim_id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, help="write one CSV row per checked claim")
    parser.add_argument("--paper-dir", type=Path, default=pn.ROOT / "paper")
    args = parser.parse_args()

    rows = []
    for name in LEDGERS:
        path = args.paper_dir / name
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    with (args.paper_dir / "known_discrepancies.csv").open(newline="", encoding="utf-8") as handle:
        known = {r["claim_id"]: r for r in csv.DictReader(handle)}

    results, mismatched = [], set()
    for row in rows:
        try:
            expected = expected_for(row)
        except Exception as exc:  # a failed recomputation is a failed claim
            expected = f"ERROR: {type(exc).__name__}: {exc}"
        ok = expected == row["printed"]
        if not ok:
            mismatched.add(row["claim_id"])
        results.append({**row, "recomputed": expected, "match": ok,
                        "known_discrepancy": row["claim_id"] in known})

    unexpected = sorted(mismatched - set(known))
    resolved = sorted(set(known) - mismatched)
    by_location: dict[str, list[int]] = {}
    for r in results:
        stats = by_location.setdefault(r["location"], [0, 0])
        stats[0] += 1
        stats[1] += int(r["match"])
    print(f"{'location':38s} {'claims':>7s} {'match':>7s}")
    for location, (total, good) in by_location.items():
        print(f"{location:38s} {total:7d} {good:7d}")
    print(f"{'TOTAL':38s} {len(results):7d} {sum(r['match'] for r in results):7d}")
    for claim_id in sorted(mismatched):
        r = next(x for x in results if x["claim_id"] == claim_id)
        tag = "known" if claim_id in known else "UNEXPECTED"
        print(f"  [{tag}] {claim_id} {r['location']}: printed {r['printed']!r}, recomputed {r['recomputed']!r}")
    for claim_id in resolved:
        print(f"  [RESOLVED] {claim_id} is listed in known_discrepancies.csv but now matches")

    if args.report:
        with args.report.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
    status = "PASS" if not unexpected and not resolved else "FAIL"
    print(f"{status}: {len(results)} claims, {len(mismatched)} mismatches "
          f"({len(mismatched) - len(unexpected)} documented in paper/known_discrepancies.csv)")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
