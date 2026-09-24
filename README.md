# SOLAR: fixed orthogonal label anchors for single-cell reference mapping

Source code, configurations, per-run results and provenance records for the
manuscript *SOLAR: Fixed orthogonal label anchors for single-cell reference
mapping* (main text and Additional file 1).

SOLAR (Supervised Orthogonal Label-Anchor Representation) assigns every
reference label a fixed orthogonal direction in the output space and trains an
expression encoder against those directions in a supervised contrastive
objective. Query cells are mapped with the reference-fitted preprocessing and
the frozen encoder.

## Check every number in the paper

```bash
python -m venv .venv && . .venv/bin/activate
pip install pandas numpy pyyaml scikit-learn
python scripts/verify_paper_numbers.py --report verification_report.csv
```

The check needs no raw expression data, GPU or model download, and runs in
under a minute. The ledger has 5,207 rows: 4,888 printed numbers and
statements that are recomputed from the per-run records in `data/` (or read
from `configs/` and the source) and compared at the printed precision, plus
319 non-result tokens (axis ticks, equation and interval notation) that are
recorded only so that the completeness check below can account for them.
Coverage by part of the paper:

| Source in the paper | What is checked |
|---|---|
| Abstract, Results, Discussion; Tables 1-3 | every reported mean, SD, count and comparison |
| Figure 2 value labels | per-dataset differences and method means (Figures 3-4 print no data values; their ticks are registered only) |
| Additional file 1, Tables S1-S9 | all 4,504 cells: values, SDs, coverage (D/B/F, [x/y]), NA codes |
| Figure S2 | the 16 confusion-matrix entries |
| Figures S3 and S5 | axis end labels and `n=` coverage labels of all 17 panels |
| Methods / Additional file 1, Section 2 | hyperparameters and protocol constants in `configs/` and the source |
| Qualitative statements | e.g. "every batch mean favoured SOLAR", "errors concentrated in CD16+ monocytes" |

The printed values are recorded in `paper/claims_*.csv`, one row per number,
with a machine-readable specification of how it is computed. Nothing is
rounded before the final comparison. The run passes only when every claim
matches, apart from entries listed in `paper/known_discrepancies.csv`, each of which
names the corresponding manuscript edit. `tools/check_ledger_coverage.py` confirms that the ledger accounts for
every numeric token in the two PDFs (228 in the main text, 5,899 in Additional
file 1). It needs the PDFs, which are not distributed here.

Additional checks:

```bash
pip install torch anndata h5py pytest matplotlib
pip install -e . --no-deps
python -m pytest tests                       # Methods behaviour: encoder, anchors, pools, Eqs. (4)-(5)
python scripts/audit_solar_numerics.py       # loss/optimizer numerical contracts
python examples/smoke_test.py                # train and map a synthetic reference/query pair on CPU
python scripts/build_paper_figures.py        # rebuild Figures 2-4 and S2-S5 into rebuilt/figures/
```

## Aggregation used throughout

For each fixed held-out batch, a stochastic method contributes a batch mean
only if all five fits (seeds 40-44) are finite. Matched PCA contributes its
single deterministic fit. Batch means are averaged with equal weight within a
dataset, then datasets are averaged with equal weight. SDs are sample SDs
(ddof=1): across the nine batch means in Tables 2 and S7, and across the five
fits in Table S3. Paired contrasts (Tables S8 and S9) use only batches complete
for both representations. Missing values are never replaced with zero. The
code is in `scripts/paper_numbers.py`.

## Repository layout

| Path | Contents |
|---|---|
| `src/SOLAR/` | SOLAR model, anchor bank, loss, training loop, benchmark adapter |
| `src/scib_benchmark/`, `src/scib_data/` | splits, reference-only preprocessing, baseline runners (Matched PCA, scVI, scANVI), official scIB scoring layer |
| `adapters/sclsc_v1_1.py` | SCLSC adapter (the upstream SCLSC source is not redistributed; see below) |
| `scripts/` | held-out experiment entry points (`experiment_v1_2.py`), scoring (`score_experiment_v1_2.py`, `paper_scib_worker.py`, `biovalid_ref_query_joint.py`), table builders, the paper verifier and the figure builder |
| `configs/` | the configurations of every reported run (see below) |
| `data/heldout/` | per-run records of the held-out comparison: `runs.csv` (378 evaluation records) and `metrics.csv` (13,608 metric rows), companion tables, and the Villani per-cell predictions behind Figure S2 |
| `data/within_batch/` | the 75 within-batch SOLAR runs behind Table S4 |
| `data/splits/` | held-out batch composition (Table S1) and the batch-selection audit |
| `data/figures/` | Figure S1 UMAP coordinates |
| `paper/` | ledgers of printed numbers and the (empty) list of known discrepancies |
| `provenance/` | source mapping and the historical execution sources (see `PROVENANCE.md`) |
| `environments/` | locked environments for SOLAR, SCLSC, analysis and the legacy official-scIB backend |

Configurations:

- `benchmark_v1_2_dim30.yaml`, `experiment_v1_2.yaml`: the common 30-dimensional
  comparison (SOLAR30, SCLSC30, Matched PCA 30D) and all evaluation settings.
- `benchmark_rebuttal_v1.yaml`: the 128-dimensional held-out runs (repeated,
  unique and no-anchor SOLAR), frozen-reference scVI/scANVI, and the 128D
  within-batch runs of Table S4. The file name is the historical run
  identifier.
- `benchmark_rebuttal_dim30.yaml`, `benchmark_rebuttal_dim64.yaml`: the 30D and
  64D within-batch runs of Table S4.
- `baseline_extension_v1_1.yaml`: the 16-dimensional SCLSC runs.
- `scib_datasets.yaml`: the three scIB Figshare files with checksums.

## Release attachments

The `v1.0` release carries the complete frozen result bundle
(`SOLAR_v1.2_Final_Experimental_Results.zip`: splits, reference-selected
genes, per-cell predictions, evaluation samples, metric provenance and
training records) and the model attachments (`SOLAR_v1.2_Model_Artifacts_part01.zip`,
`part02.zip`: embeddings, SOLAR checkpoints and PCA transforms). They are
byte-identical to the attachments of the authors' earlier (archived) repository;
their `v1.2` file names are internal identifiers. `data/heldout/metrics.csv` and
`runs.csv` are the same files as in the bundle (SHA-256 in `MANIFEST.sha256`).

Materialize and rebuild the companion tables without raw data:

```bash
python scripts/materialize_experiment_v1_2.py \
  --bundle downloads/SOLAR_v1.2_Final_Experimental_Results.zip \
  --bundle-sha256 "$(cut -d' ' -f1 downloads/SOLAR_v1.2_Final_Experimental_Results.zip.sha256)" \
  --attachment-dir downloads --output-dir release
python scripts/build_experiment_v1_2_tables.py --input-dir release/bundle --output-dir rebuilt
```

## Re-running training and scoring

Install the locked environments (Linux x86-64):

```bash
uv sync --project environments/solar --frozen --no-install-project --python 3.12
uv pip install --python environments/solar/.venv/bin/python --no-deps -e .
RPY2_CFFI_MODE=ABI uv sync --project environments/analysis --frozen --no-install-project --extra experiment --python 3.11
uv sync --project environments/sclsc-v1_1 --frozen --no-install-project --python 3.11
```

Download the three processed scIB datasets listed in `configs/scib_datasets.yaml`
(Figshare record 12420968, version 8, CC BY 4.0) into `data/scib/raw/`. The
commands `inventory`, `audit`, `prepare`, `run`, `score`, `aggregate` and
`package` of `scripts/experiment_v1_2.py` then reproduce the 30-dimensional
comparison (one registered training key per `run` call; set `mode: local` in
the config first). Fresh official-scIB scoring requires the pinned legacy
backend:

```bash
pixi install --manifest-path environments/scib-paper/pixi.toml --locked
pixi run --manifest-path environments/scib-paper/pixi.toml install-scib
pixi run --manifest-path environments/scib-paper/pixi.toml install-kbet
```

The graph-LISI helper of that backend draws its 50% subsample from a
system-generated seed, so fresh iLISI values can differ slightly from the
released ones (Additional file 1, Section 2.4). The 128D, control, scVI,
scANVI, Matched PCA 40D and within-batch runs were produced by the historical
job pipeline documented in `PROVENANCE.md`.

SCLSC is obtained separately from its authors, at the pinned commit:

```bash
git clone https://github.com/yaozhong/SCLSC.git external/SCLSC
git -C external/SCLSC checkout 2e827dfebb793dffb548fd297e5c4e70fa40692f
```

## Use SOLAR on your own data

```python
from SOLAR.benchmark import PreprocessConfig, TrainConfig, run_inductive

result = run_inductive(
    reference=adata_reference,   # AnnData with reference labels in .obs
    query=adata_query,           # query labels are never read
    variant="solar_orthogonal",
    labels_key="cell_type",
    batch_key="batch",
    seed=40,
    anchor_seed=0,
    preprocess=PreprocessConfig(n_components=40, source="X"),
    train=TrainConfig(embedding_dim=30),
)
result.save("outputs/solar_seed40")
```

## Data and licence

The processed benchmark datasets are distributed by the scIB project through
Figshare (doi:10.6084/m9.figshare.12420968.v8, CC BY 4.0); no expression
matrix is included here. Released cell IDs and predictions remain linkable to
those files. The code is released under the MIT License (`LICENSE`).
