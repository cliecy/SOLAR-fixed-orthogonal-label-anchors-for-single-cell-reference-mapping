#!/usr/bin/env python3
"""Extract the printed numbers of Additional file 1, Tables S1-S9, into ledger rows.

Maintainer tool: it reads `pdftotext -layout` output of the Additional file 1
PDF (not distributed here) and writes `paper/claims_si_tables.csv`. Every row
records the printed string exactly as typeset together with a machine-readable
specification that `scripts/verify_paper_numbers.py` recomputes from the
per-run data. The committed CSV is the reference; this tool only documents how
it was produced.

Usage:
    pdftotext -layout additional_file.pdf si.txt
    python tools/extract_si_tables.py si.txt paper/claims_si_tables.csv
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata

DATASET = {"Immune": "immune_cell_human", "Lung": "lung_atlas", "Pancreas": "pancreas",
           "Simulation 1": "simulation_1", "Simulation 2": "simulation_2"}
METHOD = {"SOLAR": "solar_orthogonal", "Matched PCA": "unintegrated_pca_matched",
          "scVI": "frozen_reference_scvi", "scANVI": "frozen_reference_scanvi",
          "SCLSC": "sclsc_refonly", "SOLAR unique": "solar_orthogonal_uniq",
          "SOLAR no anchor": "solar_none"}
S7_METHOD = {"SOLAR 128D": ("solar_orthogonal", 128), "Matched PCA (40D)": ("unintegrated_pca_matched", 40),
             "scVI 30D": ("frozen_reference_scvi", 30), "scANVI 30D": ("frozen_reference_scanvi", 30),
             "SCLSC 16D": ("sclsc_refonly", 16), "SOLAR 30D": ("solar_orthogonal", 30),
             "Matched PCA (30D)": ("unintegrated_pca_matched", 30), "SCLSC 30D": ("sclsc_refonly", 30)}
ARM = {"repeated": ("solar_orthogonal", 128), "unique": ("solar_orthogonal_uniq", 128),
       "none": ("solar_none", 128)}
CONTRAST = {"SOLAR30 minus128": (("solar_orthogonal", 30), ("solar_orthogonal", 128)),
            "SCLSC30 minus16": (("sclsc_refonly", 30), ("sclsc_refonly", 16)),
            "Repeated minus none128": (("solar_orthogonal", 128), ("solar_none", 128)),
            "Repeated minus unique128": (("solar_orthogonal", 128), ("solar_orthogonal_uniq", 128))}
METRIC = {"ARI": "ARI_cluster/label", "label ASW": "ASW_label", "batch ASW": "ASW_label/batch",
          "NMI": "NMI_cluster/label", "PCR batch": "PCR_batch", "Balanced accuracy": "balanced_accuracy",
          "cLISI": "cLISI", "Cell-cycle conservation": "cell_cycle_conservation",
          "HVG overlap": "hvg_overlap", "iLISI": "iLISI",
          "Isolated-label silhouette (source)": "isolated_label_silhouette", "Macro-F1": "macro_f1",
          "Pseudotime smoothness": "pseudotime_smoothness", "Trajectory conservation": "trajectory",
          "Trustworthiness": "trustworthiness",
          "Raw donor silhouette": "within_label_silhouette__donor",
          "Raw patient-group silhouette": "within_label_silhouette__patientGroup",
          "Raw sample-ID silhouette": "within_label_silhouette__sample_ID",
          "Raw study silhouette": "within_label_silhouette__study"}
S56_METRIC = {"Trustworthiness": "trustworthiness", "Pseudotime smoothness": "pseudotime_smoothness",
              "study silhouette": "within_label_silhouette__study",
              "sample-ID silhouette": "within_label_silhouette__sample_ID",
              "donor silhouette": "within_label_silhouette__donor",
              "patient-group silhouette": "within_label_silhouette__patientGroup"}
POP = {"J": "joint", "Q": "query", "R": "reference"}
NUM = r"-?\d+\.\d+"
MET_RE = "|".join(re.escape(k) for k in sorted(METRIC, key=len, reverse=True))

rows: list[dict] = []


def add(table: str, printed: str, kind: str, **spec) -> None:
    rows.append({"claim_id": f"{table}-{len(rows) + 1:05d}", "doc": "si", "location": table,
                 "printed": printed, "kind": kind, "spec": json.dumps(spec, sort_keys=True)})


def clean(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).replace("−", "-")
    # Drop page numbers that pdftotext places at the start of rotated-table rows.
    return re.sub(r"^\s*\d{1,2}\s{2,}(?=[A-Za-z])", "", line).strip()


def section(lines: list[str], start: str, end: str) -> list[str]:
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = next(k for k, l in enumerate(lines) if k > i and l.startswith(end))
    return lines[i:j]


def main(src: str, dst: str) -> None:
    lines = [clean(l) for l in open(src, encoding="utf-8")]

    # Table S1: composition.
    s1 = re.compile(r"^(Immune|Lung|Pancreas|Simulation [12])\s+(\S+)\s+([\d,]+)\s+([\d,]+)\s+(\d+)\s+(\d+)\s+(\d\.\d)$")
    for l in section(lines, "Table S1:", "Table S2:"):
        m = s1.match(l)
        if m:
            ds, b = DATASET[m[1]], m[2]
            for field, val in zip(("n_reference", "n_query", "n_celltypes_reference", "n_celltypes_query",
                                   "fraction_query_cells_label_supported_by_reference"), m.groups()[2:]):
                add("Table S1", val, "composition", dataset=ds, batch=b, field=field)

    # Table S3: per-batch annotation.
    s3 = re.compile(rf"^(A|B)\s+(Immune|Lung|Pancreas) / (\S+)\s+(SOLAR|Matched PCA|scVI|scANVI|SCLSC) \((\d+)D\)\s+(\d)\s+({NUM})\s+(?:± ({NUM})|\(SD –\))\s+({NUM})\s+(?:± ({NUM})|\(SD –\))$")
    for l in section(lines, "Table S3:", "Table S4:"):
        m = s3.match(l)
        if m:
            block, ds, b, meth, dim, n, f1, f1sd, ba, basd = m.groups()
            common = dict(block=block, method=METHOD[meth], dim=int(dim), dataset=DATASET[ds], batch=b)
            add("Table S3", n, "batch", stat="n", metric="macro_f1", population="query", **common)
            add("Table S3", f1, "batch", stat="mean", metric="macro_f1", population="query", **common)
            add("Table S3", ba, "batch", stat="mean", metric="balanced_accuracy", population="query", **common)
            for sd, met in ((f1sd, "macro_f1"), (basd, "balanced_accuracy")):
                add("Table S3", sd if sd else "–", "batch", stat="seed_sd", metric=met, population="query", **common)

    # Table S4: within-batch output width.
    s4 = re.compile(r"^(Immune|Lung|Pancreas|Simulation [12])\s+(30|64|128)\s+" + r"\s+".join([r"(\d\.\d{3}|–)"] * 6) + "$")
    names = ["ASW_label", "NMI_cluster/label", "ARI_cluster/label", "iLISI", "PCR_batch", "trajectory"]
    for l in section(lines, "Table S4:", "Table S5:"):
        m = s4.match(l)
        if m:
            add("Table S4", m[2], "width_dim", dataset=DATASET[m[1]], dim=int(m[2]))
            for met, val in zip(names, m.groups()[2:]):
                add("Table S4", val, "width", dataset=DATASET[m[1]], dim=int(m[2]), metric=met)

    # Tables S5 and S6: dataset-level local-structure and metadata panels.
    cell = r"(NA|-?\d\.\d{3}) \[(\d)/(\d)\]"
    s56 = re.compile(rf"^(Immune|Lung) / ({'|'.join(S56_METRIC)})\s+(A|B|Control) / (SOLAR no anchor|SOLAR unique|SOLAR|Matched PCA|scVI|scANVI|SCLSC) \((\d+)D\)\s+{cell}\s+{cell}\s+{cell}\s+(–|P|C)$")
    for table, start, end in (("Table S5", "Table S5:", "Table S6:"), ("Table S6", "Table S6:", "Table S7:")):
        for l in section(lines, start, end):
            m = s56.match(l)
            if not m:
                continue
            g = m.groups()
            common = dict(block=g[2], method=METHOD[g[3]], dim=int(g[4]), dataset=DATASET[g[0]], metric=S56_METRIC[g[1]])
            for k, pop in enumerate(("reference", "query", "joint")):
                val, valid, applicable = g[5 + 3 * k: 8 + 3 * k]
                add(table, val, "dataset", stat="mean", population=pop, **common)
                add(table, f"{valid}/{applicable}", "dataset", stat="coverage", population=pop, **common)
            add(table, g[14], "dataset", stat="na_code", population="all", **common)

    # Table S7 and the individual-arm part of Table S9: pooled summaries.
    meth_re = "|".join(re.escape(k) for k in S7_METHOD)
    val = rf"(?:({NUM}) ± ({NUM})|({NUM}|NA) \(SD –\))"
    cov = r"(\d/\d)\s+(\d/\d)\s+(\d+/\d+)\s+(\S+)"
    s7 = re.compile(rf"^(A|B)\s+({meth_re})\s+({MET_RE}) / (J|Q|R)\s+{val}\s+{cov}$")
    s9arm = re.compile(rf"^(repeated|unique|none)\s+({MET_RE}) / (J|Q|R)\s+{val}\s+{cov}$")
    for table, start, end, pattern in (("Table S7", "Table S7:", "Table S8:", s7),
                                       ("Table S9 (arms)", "Table S9: (continued). Full E3", "Supplementary figures", s9arm)):
        for l in section(lines, start, end):
            m = pattern.match(l)
            if not m:
                continue
            g = m.groups()
            if table == "Table S7":
                method, dim = S7_METHOD[g[1]]
                common = dict(block=g[0], method=method, dim=dim, metric=METRIC[g[2]], population=POP[g[3]])
                g = g[4:]
            else:
                method, dim = ARM[g[0]]
                common = dict(method=method, dim=dim, metric=METRIC[g[1]], population=POP[g[2]])
                g = g[3:]
            mean, sd, single = g[0], g[1], g[2]
            add(table, mean if mean else single, "summary", stat="mean", **common)
            add(table, sd if sd else "–", "summary", stat="sd", **common)
            for stat, text in zip(("D", "B", "F", "na_code"), g[3:]):
                add(table, text, "summary", stat=stat, **common)

    # Tables S8 and S9: paired contrasts.
    num_or_na = rf"({NUM}|NA)"
    delta = r"([+-]?\d+\.\d+|NA)"
    s89 = re.compile(rf"^({'|'.join(CONTRAST)})\s+({MET_RE}) / (J|Q|R)\s+{num_or_na}\s+{num_or_na}\s+{delta}\s+(\d/\d)\s+(\d/\d)\s+(\d+/\d+)\s+(\d+/\d+)$")
    for table, start, end in (("Table S8", "Table S8:", "Table S9:"), ("Table S9", "Table S9:", "Table S9: (continued). Full E3")):
        for l in section(lines, start, end):
            m = s89.match(l)
            if not m:
                continue
            g = m.groups()
            left, right = CONTRAST[g[0]]
            common = dict(left=list(left), right=list(right), metric=METRIC[g[1]], population=POP[g[2]])
            for stat, text in zip(("L", "R", "delta", "D", "B", "FL", "FR"), g[3:]):
                add(table, text, "contrast", stat=stat, **common)

    with open(dst, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["location"]] = counts.get(row["location"], 0) + 1
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
