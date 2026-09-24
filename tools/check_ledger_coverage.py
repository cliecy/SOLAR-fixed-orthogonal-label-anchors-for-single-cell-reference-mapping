#!/usr/bin/env python3
"""Completeness check: every number printed in the two PDFs must be in the ledger.

Maintainer tool; the PDFs are not distributed with this repository.

    pdftotext -layout bmc_main.pdf main.txt
    pdftotext -layout additional_file.pdf si.txt
    python tools/check_ledger_coverage.py main.txt si.txt

Numeric tokens are collected from the body text after removing manuscript line
numbers, page numbers, citation brackets, cross-references (Table 2, Figure S3,
Eq. (4), Section 3.5, Additional file 1), the title page, the Declarations and
the reference lists. The multiset of remaining tokens must be contained in the
multiset of tokens of the ledger's printed strings for the same document. The
tool prints every uncovered token with its line for review and exits non-zero
if any remain.
"""
from __future__ import annotations

import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN = re.compile(r"\d+(?:[.,]\d+)*")
CROSS_REF = re.compile(
    r"\b(?:Tables?|Figures?|Figs?\.|Eqs?\.|Equations?|Sections?|Algorithm|Additional file|Additional ﬁle)"
    r"\s*\(?S?\d+(?:\.\d+)?[A-J]?\)?(?:\s*(?:and|–|-|,)\s*S?\d+[A-J]?)*"
    r"|\bS\d+[A-J]?\b|\(\d\)|\[\d+(?:\s*[–,-]\s*\d+)*\]")


def normalize(line: str) -> str:
    return unicodedata.normalize("NFKC", line).replace("−", "-")


def main_body(lines: list[str]) -> list[tuple[int, str]]:
    start = next(i for i, l in enumerate(lines) if "Abstract" in l)
    stop = next(i for i, l in enumerate(lines) if re.search(r"\bDeclarations\b", l))
    out = []
    for i in range(start, stop):
        line = normalize(lines[i])
        line = re.sub(r"^\s*\d{3,4}(?=\s|$)", " ", line)            # left-margin line number
        line = re.sub(r"(?<![\d.,])\d{3,4}\s*$", " ", line)          # right-margin line number
        if re.fullmatch(r"\s*\d{1,2}\s*", line):                     # page number
            continue
        line = re.sub(r"^\s*\d(?:\.\d)?\s+(?=[A-Z])", " ", line)       # section heading number
        out.append((i + 1, line))
    return out


def si_body(lines: list[str]) -> list[tuple[int, str]]:
    start = next(i for i, l in enumerate(lines) if l.startswith("1     Supplementary study details"))
    stop = max(i for i, l in enumerate(lines) if l.strip() == "References")
    out = []
    for i in range(start, stop):
        line = normalize(lines[i])
        line = re.sub(r"^\s*\d{1,2}\s{2,}(?=[A-Za-z])", " ", line)   # rotated-page numbers
        if re.fullmatch(r"\s*\d{1,2}\s*", line):
            continue
        line = re.sub(r"^\s*\d(?:\.\d)?\s{2,}(?=[A-Z])", " ", line)   # section numbers
        out.append((i + 1, line))
    return out


IDENTIFIER = re.compile(r"(?<=[A-Za-z_+])\d+(?:\.\d+)?|\d+(?:\.\d+)?(?=[A-Za-z_+])"
                        r"|\bLung\s*[:/]\s*\d\b|\bSimulation \d\b|\bFluidigm C1\b|\binDrop \d\b")


def tokens(text: str) -> list[str]:
    """Numeric tokens, ignoring cross-references and identifiers such as 10X, inDrop3, B1, 30D, F1."""
    text = CROSS_REF.sub(" ", text)
    text = IDENTIFIER.sub(" ", text)
    return [t.replace(",", "") for t in TOKEN.findall(text)]


def ledger_tokens(doc: str) -> Counter:
    counter: Counter = Counter()
    for path in sorted((ROOT / "paper").glob("claims_*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["doc"] == doc:
                    counter.update(t.lstrip("0") or "0" for t in tokens(normalize(row["printed"]).lstrip("+-")))
    return counter


def check(doc: str, body: list[tuple[int, str]]) -> int:
    available = ledger_tokens(doc)
    missing = []
    for number, line in body:
        for token in tokens(line):
            key = token.lstrip("0") or "0"
            if available[key] > 0:
                available[key] -= 1
            else:
                missing.append((number, token, line.strip()))
    for number, token, line in missing:
        print(f"{doc} line {number}: uncovered {token!r}: {line[:120]}")
    print(f"{doc}: {sum(len(tokens(l)) for _, l in body)} numeric tokens, {len(missing)} uncovered")
    return len(missing)


def main(main_txt: str, si_txt: str) -> int:
    main_lines = Path(main_txt).read_text(encoding="utf-8").splitlines()
    si_lines = Path(si_txt).read_text(encoding="utf-8").splitlines()
    uncovered = check("main", main_body(main_lines)) + check("si", si_body(si_lines))
    return 1 if uncovered else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
