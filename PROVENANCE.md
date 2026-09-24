# Provenance

This repository (release v1.0) is the publication package for the manuscript.
Its numerical content is identical to the final state of the authors' earlier
working repository, which is archived privately; nothing here was retrained or
rescored. File names that contain `v1_1` or `v1_2` (scripts, configs, release
attachments) are internal experiment identifiers of that earlier work. They are
kept unchanged because recorded hashes and the bundle indexes refer to them.

## Where each reported result comes from

| Result | Runs | Produced by |
|---|---|---|
| Common 30-dimensional comparison (Tables 2-3, S3 block B, S5-S7 block B; Figs. 2, 4, S4) | SOLAR30 and SCLSC30: 90 trainings (9 batches x 5 seeds each); Matched PCA 30D: 9 deterministic fits | `scripts/experiment_v1_2.py` with `configs/benchmark_v1_2_dim30.yaml` and `configs/experiment_v1_2.yaml` |
| 128-dimensional SOLAR and anchor controls (Tables S5-S9; Figs. 3, S1-S3) | repeated, unique and no-anchor SOLAR, 45 runs each | historical job pipeline, `configs/benchmark_rebuttal_v1.yaml` |
| scVI and scANVI (30D frozen reference) | 45 runs each | historical job pipeline, `configs/benchmark_rebuttal_v1.yaml` (`baselines.scvi`, `baselines.scanvi`) |
| Matched PCA 40D | 9 deterministic fits | `provenance/historical_execution/scripts/build_matched_unintegrated.py` |
| SCLSC 16D | 45 runs | `adapters/sclsc_v1_1.py`, `configs/baseline_extension_v1_1.yaml` |
| Within-batch output width (Table S4) | SOLAR 30/64/128D, 5 datasets x 5 split seeds | historical job pipeline, `configs/benchmark_rebuttal_{v1,dim30,dim64}.yaml` |
| Held-out batch composition and selection (Tables 1, S1) | 57 possible held-out batches, 15 selected | `provenance/historical_execution/scripts/build_heldout_selection_audit.py` |

All held-out evaluation records (annotation readout, official scIB metrics,
trustworthiness, pseudotime smoothness, within-label silhouettes) were
assembled by `scripts/score_experiment_v1_2.py` into `data/heldout/metrics.csv`.
Official scIB values come from the pinned scIB commit
`e2a37e0ed63dc34b60aa535cc656400552af757a` run by `scripts/paper_scib_worker.py`.
Historical official records were reused by semantic identity rather than
recomputed. Each record's raw worker output, job specification and backend
identity are in `metric_provenance/` of the release bundle.

## Historical execution sources

The 128D SOLAR, control, scVI/scANVI, Matched PCA 40D and within-batch runs
predate the v1.2 public layout. The files under `provenance/historical_execution/`
are byte copies of the job builder, workers and helper scripts that ran them,
kept for disclosure and not imported by the current code:

- `src/scib_benchmark/jobs.py`, `solar_worker.py`, `baseline_runners.py`;
- `scripts/build_benchmark_jobs.py`, `prepare_benchmark_splits.py`,
  `run_benchmark_job.py`, `run_baseline_job.py`, `build_matched_unintegrated.py`,
  `register_baseline_embedding.py`, `build_heldout_selection_audit.py`;
- `solar_training_source_historical.patch`: applied to `src/SOLAR/`, it recreates
  the SOLAR training source used for the 128D runs (archive SHA-256
  `c6f0153e78eb8c65a99fd43610e5cbfac96f4d8101ac86340b8ac02fa33656d1`, recorded
  in `configs/benchmark_rebuttal_v1.yaml`).

The current `src/` differs from these files in three ways:

1. **Loss numerics.** The historical loss computed `log(0)` and multiplied it
   by zero for a view with no other view in its pool.
2. **Training loop.** The historical loop stepped the optimizer on a no-anchor
   mini-batch whose labels were all distinct; the current loop skips it.
3. **Recorded metadata.** In the historical scANVI metadata,
   `query_labels_visible_*` was recorded incorrectly as true. The query labels
   were in fact replaced by `Unknown` before frozen encoding.

The v1.2 audit (`audit/correction_decision.json` and `audit/impact.json` in
the release bundle) checked every historical held-out anchor and no-anchor
run: every mini-batch, including train and validation tails, exceeded the
number of reference labels. Neither numerical case occurred, so no historical
run was affected and none was retrained. Paths such as
`outputs/benchmark/solar_scib_rebuttal_v1/...` and URLs to the earlier or
private workspaces inside these files and inside `configs/` are historical
identifiers; they are not needed to use this repository.

## Files taken from the author's working repositories

Besides the historical execution sources above, the following were added from
the authors' non-public working repositories because the manuscript relies on
them:

- `data/splits/heldout_batch_composition.csv`,
  `data/splits/heldout_batch_selection_audit.csv` (Tables 1 and S1, batch
  selection rule);
- `data/figures/figS1_heldout_umap_coords.csv.gz` and
  `scripts/figure_s1/` (Figure S1);
- `data/heldout/villani_predictions/` (five per-cell prediction files, also
  in the release bundle, used for Figure S2).

`data/within_batch/solar_width_runs.csv` contains the 75 rows of the earlier repository's
`raw_scib_metrics_all_runs.csv` (SHA-256
`d0e65f5b2a03f77525a54e82281e5ac69c05596a865cb518acfde07ccdeba97a`) with
`method == solar_orthogonal`, `track == core` and `run` in `primary_128d`,
`dim30`, `dim64`. It was produced by selecting whole lines, so every byte of
every row is unchanged.

`source_file_mapping.json` maps each public source file to
its original source and hash.

## Figures

Figure 1 is a schematic. Figure S1 is rebuilt by `scripts/figure_s1/` from
embeddings in the model attachments. Its coordinates are in `data/figures/`.
`scripts/build_paper_figures.py` rebuilds the quantitative content of Figures
2-4 and S2-S5. Their signed-difference axes extend 16% (S3) or 17% (S5) of the
zero-inclusive data range beyond it, which reproduces the printed axis end
labels.
