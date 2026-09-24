#!/usr/bin/env python
from __future__ import print_function

import json
import os
import platform
import subprocess

import anndata
import h5py
import louvain
import numpy
import pandas
import rpy2
import scanpy
import scipy
import sklearn
import scIB


EXPECTED = {
    "python": "3.7.12",
    "numpy": "1.18.1",
    "pandas": "0.25.3",
    "scipy": "1.4.1",
    "scikit_learn": "0.22.2.post1",
    "scanpy": "1.4.6",
    "anndata": "0.7.1",
    "h5py": "2.10.0",
    "louvain": "0.6.1",
    "rpy2": "3.1.0",
}
SCIB_COMMIT = "e2a37e0ed63dc34b60aa535cc656400552af757a"
KBET_COMMIT = "afc5f431bcbefd73267acc066a0f2e4eaa10a355"


def main():
    observed = {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "scanpy": scanpy.__version__,
        "anndata": anndata.__version__,
        "h5py": h5py.__version__,
        "louvain": str(getattr(louvain, "__version__", "unknown")),
        "rpy2": rpy2.__version__,
    }
    observed = {key: str(value) for key, value in observed.items()}
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in EXPECTED.items()
        if observed.get(key) != value
    }
    prefix = os.environ.get("CONDA_PREFIX", "")
    marker = os.path.join(prefix, "conda-meta", "scib-paper-source.txt")
    marker_commit = None
    if os.path.isfile(marker):
        with open(marker, "r") as handle:
            marker_commit = handle.readline().strip()
    if marker_commit != SCIB_COMMIT:
        mismatches["scib_commit"] = {
            "expected": SCIB_COMMIT,
            "observed": marker_commit,
        }
    kbet_marker = os.path.join(prefix, "conda-meta", "kbet-source.txt")
    kbet_commit = None
    if os.path.isfile(kbet_marker):
        with open(kbet_marker, "r") as handle:
            kbet_commit = handle.readline().strip()
    if kbet_commit != KBET_COMMIT:
        mismatches["kbet_commit"] = {
            "expected": KBET_COMMIT,
            "observed": kbet_commit,
        }
    lisi_binary = os.path.join(os.path.dirname(scIB.__file__), "knn_graph", "knn_graph.o")
    if not os.path.isfile(lisi_binary) or not os.access(lisi_binary, os.X_OK):
        mismatches["lisi_binary"] = {
            "expected": "present and executable from the fixed scIB commit",
            "observed": lisi_binary,
        }
    r_version = subprocess.check_output(
        ["Rscript", "-e", "cat(as.character(getRversion()))"],
        universal_newlines=True,
    ).strip()
    kbet_version = subprocess.check_output(
        [
            "Rscript",
            "-e",
            "stopifnot(requireNamespace('kBET', quietly=TRUE)); cat(as.character(packageVersion('kBET')))",
        ],
        universal_newlines=True,
    ).strip()
    payload = {
        "backend": "paper_scib_0_2_0",
        "scib_commit": marker_commit,
        "r_version": r_version,
        "kbet_version": kbet_version,
        "kbet_commit": kbet_commit,
        "lisi_binary": lisi_binary,
        "packages": observed,
        "mismatches": mismatches,
        "state": "passed" if not mismatches else "failed",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
