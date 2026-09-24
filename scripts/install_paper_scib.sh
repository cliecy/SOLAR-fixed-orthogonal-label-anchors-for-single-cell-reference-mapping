#!/usr/bin/env bash
set -euo pipefail

SCIB_REPOSITORY="https://github.com/theislab/scib.git"
SCIB_COMMIT="e2a37e0ed63dc34b60aa535cc656400552af757a"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Run this script through the paper-scib Pixi environment." >&2
  exit 2
fi

marker="${CONDA_PREFIX}/conda-meta/scib-paper-source.txt"
package_dir="$(python -c 'import os, scIB; print(os.path.dirname(scIB.__file__))' 2>/dev/null || true)"
if [[ -f "$marker" ]] && [[ "$(sed -n '1p' "$marker")" == "$SCIB_COMMIT" ]] \
    && [[ -x "$package_dir/knn_graph/knn_graph.o" ]]; then
  python -c "import scIB" >/dev/null
  echo "scIB paper source already installed at ${SCIB_COMMIT}"
  exit 0
fi

task_temp="$(mktemp -d -t scib-paper-XXXXXXXX)"
trap 'rm -rf "$task_temp"' EXIT
git clone --quiet --no-checkout "$SCIB_REPOSITORY" "$task_temp/scib"
git -C "$task_temp/scib" checkout --quiet "$SCIB_COMMIT"
observed_commit="$(git -C "$task_temp/scib" rev-parse HEAD)"
if [[ "$observed_commit" != "$SCIB_COMMIT" ]]; then
  echo "scIB checkout mismatch: expected $SCIB_COMMIT, observed $observed_commit" >&2
  exit 1
fi
python -m pip install --no-deps "$task_temp/scib"
package_dir="$(python -c 'import os, scIB; print(os.path.dirname(scIB.__file__))')"
install -d "$package_dir/knn_graph"
install -m 0755 "$task_temp/scib/scIB/knn_graph/knn_graph.o" "$package_dir/knn_graph/knn_graph.o"
python -c "import scIB; from pathlib import Path; assert (Path(scIB.__file__).parent / 'knn_graph' / 'knn_graph.o').is_file()"
printf '%s\n%s\n' "$SCIB_COMMIT" "$SCIB_REPOSITORY" > "$marker"
echo "Installed scIB paper source at ${SCIB_COMMIT}"
