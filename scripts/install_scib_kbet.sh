#!/usr/bin/env bash
set -euo pipefail

KBET_REPOSITORY="https://github.com/theislab/kBET.git"
KBET_COMMIT="afc5f431bcbefd73267acc066a0f2e4eaa10a355"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Run this script inside the intended Pixi environment." >&2
  exit 2
fi
marker="${CONDA_PREFIX}/conda-meta/kbet-source.txt"
if [[ -f "$marker" ]] && [[ "$(sed -n '1p' "$marker")" == "$KBET_COMMIT" ]] \
    && Rscript -e "stopifnot(requireNamespace('kBET', quietly=TRUE))" >/dev/null 2>&1; then
  echo "kBET already installed at ${KBET_COMMIT}"
  exit 0
fi
TASK_TEMP="$(mktemp -d -t scib-kbet-XXXXXXXX)"
trap 'rm -rf "$TASK_TEMP"' EXIT

git clone --quiet --no-checkout "$KBET_REPOSITORY" "$TASK_TEMP/kBET"
git -C "$TASK_TEMP/kBET" checkout --quiet "$KBET_COMMIT"
observed_commit="$(git -C "$TASK_TEMP/kBET" rev-parse HEAD)"
if [[ "$observed_commit" != "$KBET_COMMIT" ]]; then
  echo "kBET checkout mismatch: expected $KBET_COMMIT, observed $observed_commit" >&2
  exit 1
fi
R CMD INSTALL "$TASK_TEMP/kBET"
Rscript -e "stopifnot(requireNamespace('kBET', quietly=TRUE)); cat('kBET ', as.character(packageVersion('kBET')), '\n', sep='')"
printf '%s\n%s\n' "$KBET_COMMIT" "$KBET_REPOSITORY" > "$marker"
