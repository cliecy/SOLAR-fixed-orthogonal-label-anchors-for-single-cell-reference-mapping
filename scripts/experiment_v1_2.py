#!/usr/bin/env python3
"""Portable released-bundle verification and separately identified local reproduction."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import yaml

from materialize_experiment_v1_2 import member_path, sha, verify_directory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def resolve(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def interpreter(value):
    # Executing the resolved symlink target bypasses pyvenv.cfg and its packages.
    path = Path(value)
    return path.absolute() if path.is_absolute() else (ROOT / path).absolute()


def execute(argv):
    subprocess.run([str(x) for x in argv], check=True, cwd=ROOT,
                   env={**os.environ, 'PYTHONPATH': str(ROOT / 'src'),
                        'CUBLAS_WORKSPACE_CONFIG': ':4096:8'})


def source_hashes():
    paths = [p for base in ('src', 'scripts', 'adapters') for p in (ROOT / base).rglob('*.py')]
    paths += [ROOT / 'configs/benchmark_v1_2_dim30.yaml']
    paths += [ROOT / 'environments' / name / lock for name, lock in
              [('solar', 'uv.lock'), ('analysis', 'uv.lock'), ('sclsc-v1_1', 'uv.lock'), ('scib-paper', 'pixi.lock')]]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


class Bundle:
    def __init__(self, config):
        self.root = resolve(config['bundle_dir'])
        self.attachments = resolve(config['attachment_dir'])
        bootstrap = load(resolve(config['bootstrap_record']))
        if sha(self.root / 'MANIFEST.sha256') != bootstrap['bundle_manifest_sha256']:
            raise ValueError('Extracted bundle manifest changed after ZIP checksum verification')
        self.files = verify_directory(self.root)
        self.members = {}
        for part in load(self.root / 'model_artifacts.json'):
            if part not in bootstrap['attachments']:
                raise ValueError('Attachment index differs from verified bootstrap record')
            name = part['filename']
            if member_path(name).name != name:
                raise ValueError('Attachment filename must be a basename')
            directory = self.attachments / name
            if sha(directory / 'MANIFEST.sha256') != bootstrap['attachment_manifest_sha256'][name]:
                raise ValueError('Extracted attachment manifest changed after ZIP checksum verification')
            records = verify_directory(directory)
            if set(records) != set(part['members']):
                raise ValueError('Attachment member listing differs from authenticated bundle index')
            for member in records:
                if member in self.members:
                    raise ValueError(f'Ambiguous attachment member: {member}')
                self.members[member] = directory / member
        self.execution = load(self.root / 'provenance/execution_manifest.json')
        self.analysis = load(self.root / 'provenance/analysis_manifest.json')
        self.publication = load(self.root / 'publication_provenance.json')
        self.manifest_sha256 = sha(self.root / 'MANIFEST.sha256')

    def file(self, name):
        member_path(name)
        if name not in self.files:
            raise FileNotFoundError(f'Authenticated final bundle must contain {name}')
        return self.root / name

    def model(self, name):
        member_path(name)
        if name not in self.members:
            raise FileNotFoundError(f'Named model attachments must contain {name}')
        return self.members[name]

    def records(self, config):
        records = self.analysis.get('evaluation_runs', self.execution['evaluation_runs'])
        datasets = yaml.safe_load(self.file('configs/scib_datasets.yaml').read_text())['datasets']
        result = []
        for original in records:
            record = dict(original)
            run_id = str(record['run_id'])
            member_path(run_id)
            if '/' in run_id:
                raise ValueError('Run IDs must be single path components')
            dataset, batch = record['dataset_id'], str(record['heldout_batch'])
            member_path(dataset)
            member_path(batch)
            item = datasets[dataset]
            filename = item['filename']
            if member_path(filename).name != filename:
                raise ValueError('Raw dataset filename must be a basename')
            record.update(dataset_path=str(resolve(config['data_dir']) / filename),
                          dataset_sha256=item['checksum_sha256'],
                          split_path=str(self.file(f'shared_splits/{dataset}/heldout_{batch}/outer.csv')),
                          features_path=str(self.file(f'preprocessing/{dataset}/heldout_{batch}/hvg2000.txt')),
                          embedding_path=str(self.model(f'models/{run_id}/embedding.npz')),
                          output_dir=str(resolve(config['output_root']) / 'local_runs' / run_id))
            for path, digest in [('split_path', 'split_sha256'), ('features_path', 'features_sha256'),
                                 ('embedding_path', 'embedding_sha256')]:
                observed = sha(record[path])
                if record.get(digest) and record[digest] != observed:
                    raise ValueError(f'Unmodified numerical input hash mismatch: {run_id}/{path}')
                record[digest] = observed
            result.append(record)
        ids = [r['run_id'] for r in result]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate registered run IDs')
        return result


def training_records(bundle, records):
    by_id = {r['run_id']: r for r in records}
    selected = []
    keys = set()
    for original in bundle.execution['runs']:
        r = dict(by_id[original['run_id']])
        r['training_method'] = original['training_method']
        key = (r['training_method'], r['dataset_id'], str(r['heldout_batch']), int(r['seed']))
        if key in keys:
            raise ValueError('Duplicate registered training key')
        keys.add(key)
        selected.append(r)
    expected = {(method, dataset, batch, seed) for method in ('solar30', 'sclsc30')
                for dataset, batches in {'immune_cell_human': ['10X', 'Oetjen_A', 'Villani'],
                                        'lung_atlas': ['3', '4', 'B1'],
                                        'pancreas': ['fluidigmc1', 'inDrop2', 'inDrop3']}.items()
                for batch in batches for seed in range(40, 45)}
    if keys != expected or any(int(r['dimension']) != 30 for r in selected):
        raise ValueError('Released training manifest must contain exactly the registered 90 30D keys')
    return selected


def inventory(config, bundle, output, args):
    records = bundle.records(config)
    selected = training_records(bundle, records)
    save(output / 'inventory.json', {'scope': 'authenticated_released_artifacts_not_local_training',
        'bundle_manifest_sha256': bundle.manifest_sha256, 'evaluation_records': len(records),
        'registered_training_keys': len(selected), 'records': records})
    print(json.dumps({'evaluation_records': len(records), 'registered_training_keys': len(selected)}))


def audit(config, bundle, output, args):
    records = bundle.records(config)
    training_records(bundle, records)
    from scib_benchmark.audit import audit_official_scib_score
    for record in records:
        directory = f"metric_provenance/{record['run_id']}"
        spec = load(bundle.file(directory + '/job.json'))
        errors = audit_official_scib_score(spec, bundle.file(directory + '/raw.json'))
        if errors:
            raise ValueError(f"{record['run_id']}: {errors}")
        expected = {'dataset_id': record['dataset_id'], 'dataset_sha256': record['dataset_sha256'],
                    'variant': record['method'], 'seed': int(record['seed'])}
        if any(spec.get(k) != v for k, v in expected.items()):
            raise ValueError('Released official job semantic identity differs from evaluation record')
    execute([interpreter(config['solar_python']), ROOT / 'scripts/check_method_environment.py', '--method', 'solar'])
    execute([interpreter(config['solar_python']), ROOT / 'scripts/audit_solar_numerics.py'])
    save(output / 'audit.json', {'state': 'passed', 'bundle_manifest_sha256': bundle.manifest_sha256,
        'source_hashes': source_hashes(), 'scope': ['released_official_record_semantic_audit',
        'released_input_byte_identity', 'current_public_SOLAR_numerical_regression'],
        'not_claimed': ['independent_reexecution_of_historical_training', 'local_training_completion',
                        'new_local_run_leakage_perturbation_experiment']})


def prepare(config, bundle, output, args):
    audit_record = load(output / 'audit.json')
    if audit_record.get('state') != 'passed' or audit_record.get('bundle_manifest_sha256') != bundle.manifest_sha256 or audit_record.get('source_hashes') != source_hashes():
        raise ValueError('Run audit with the same bundle and current public source before prepare')
    from score_experiment_v1_2 import read_split
    records = bundle.records(config)
    selected = training_records(bundle, records)
    seen = set()
    for record in selected:
        if record['dataset_id'] not in seen:
            if sha(record['dataset_path']) != record['dataset_sha256']:
                raise ValueError(f"Raw data SHA-256 mismatch: {record['dataset_path']}")
            seen.add(record['dataset_id'])
        key = (record['dataset_id'], record['heldout_batch'])
        if key not in seen:
            read_split(record)
            seen.add(key)
    save(output / 'prepared.json', {'state': 'prepared_not_trained', 'source_hashes': source_hashes(),
        'bundle_manifest_sha256': bundle.manifest_sha256, 'runs': selected,
        'local_identity': 'new_local_reproduction_not_the_released_execution'})
    print('Prepared exactly 90 registered keys; no training has run.')


def training_predictions(embedding, reference_labels, query_ids):
    """Label-free query readout; final truth-dependent metrics belong to score."""
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    n_reference = len(reference_labels)
    if n_reference < 15 or len(embedding) != n_reference + len(query_ids):
        raise ValueError('Invalid full-reference/query prediction dimensions')
    classifier = KNeighborsClassifier(n_neighbors=15, weights='distance', metric='minkowski', p=2, n_jobs=1)
    classifier.fit(embedding[:n_reference], reference_labels)
    return pd.DataFrame({'barcode': query_ids, 'prediction': classifier.predict(embedding[n_reference:])})


def run(config, bundle, output, args):
    if config['mode'] != 'local':
        raise ValueError('Set mode: local in your config to authorize new local training; released artifacts are never overwritten')
    prepared = load(output / 'prepared.json')
    if prepared['source_hashes'] != source_hashes() or prepared['bundle_manifest_sha256'] != bundle.manifest_sha256:
        raise ValueError('Source or bundle changed after prepare')
    candidates = [r for r in prepared['runs'] if (r['training_method'], r['dataset_id'], str(r['heldout_batch']), int(r['seed'])) == (args.method, args.dataset, args.batch, args.seed)]
    if len(candidates) != 1:
        raise ValueError('run requires exactly one registered --method --dataset --batch --seed key')
    r = candidates[0]
    for path, digest in [('dataset_path', 'dataset_sha256'), ('split_path', 'split_sha256'), ('features_path', 'features_sha256')]:
        if sha(r[path]) != r[digest]:
            raise ValueError(f'Prepared input changed: {path}')
    destination = Path(r['output_dir'])
    if destination.exists():
        raise FileExistsError('Local run already has output; preserve it and select a new output_root for another attempt')
    from score_experiment_v1_2 import read_split, read_embedding
    split, _ = read_split(r)
    if args.method == 'solar30':
        spec = load(bundle.file(f"runs/{r['run_id']}/run_config.json"))
        budget = output / 'local_inputs' / r['run_id'] / 'label_budget.csv'
        budget.parent.mkdir(parents=True, exist_ok=True)
        reference = split.loc[split.role == 'reference', ['barcode']].copy()
        reference['labeled'] = True
        reference.to_csv(budget, index=False)
        spec.update(dataset_path=r['dataset_path'], split_file=r['split_path'], gene_file=r['features_path'],
            label_budget_file=str(budget), label_budget_sha256=sha(budget),
            output_dir=str(destination), solar_package_root=str(ROOT / 'src/SOLAR'),
            split_sha256=r['split_sha256'], features_sha256=r['features_sha256'],
            solar_upstream_archive='source_snapshot://execution_SOLAR',
            execution_profile='public_local_reproduction', contract_version='v1.2', save_model=True)
        expected_train = yaml.safe_load(resolve(config['solar_config']).read_text())['solar']['train']
        if spec['train'] != expected_train:
            raise ValueError('Released training settings differ from the public frozen template')
        spec_path = output / 'local_inputs' / r['run_id'] / 'job.json'
        save(spec_path, spec)
        execute([interpreter(config['solar_python']), ROOT / 'scripts/check_method_environment.py', '--method', 'solar'])
        argv = [interpreter(config['solar_python']), '-m', 'scib_benchmark.solar_worker', '--job-spec', spec_path]
    else:
        argv = [interpreter(config['sclsc_python']), ROOT / 'adapters/sclsc_v1_1.py', '--contract-version', 'v1.2',
                '--output-dim', '30', '--input', r['dataset_path'], '--split', r['split_path'],
                '--features', r['features_path'], '--output-dir', destination, '--dataset-id', r['dataset_id'],
                '--heldout-batch', r['heldout_batch'], '--seed', r['seed'], '--batch-key', r['batch_key'],
                '--label-key', r['label_key'], '--expected-reference', int(split.role.eq('reference').sum()),
                '--expected-query', int(split.role.eq('query').sum())]
    attempt = {'state': 'running', 'identity': 'new_local_training', 'run_id': r['run_id'],
               'argv': list(map(str, argv)), 'started_unix': time.time(),
               'prepared_sha256': sha(output / 'prepared.json')}
    attempt_path = output / 'attempts' / (r['run_id'] + '.json')
    save(attempt_path, attempt)
    try:
        execute(argv)
        r['embedding_path'] = str(destination / 'embedding.npz')
        embedding = read_embedding(r, split)
        predictions = training_predictions(embedding, split.loc[split.role == 'reference', 'label'].to_numpy(str),
                                           split.loc[split.role == 'query', 'barcode'].to_numpy(str))
        predictions.to_csv(destination / 'query_predictions.csv', index=False)
        names = ['embedding.npz', 'internal_reference_split.csv', 'training_history.json', 'run_info.json', 'status.json', 'query_predictions.csv']
        names += ['model/solar_model.pt', 'model/solar_metadata.json', 'pca_preprocessor.npz'] if args.method == 'solar30' else ['checkpoint.pt', 'runtime.json']
        save(destination / 'local_contract.json', {'identity': 'new_local_training', 'run_id': r['run_id'],
            'prepared_sha256': attempt['prepared_sha256'], 'artifacts': {n: sha(destination / n) for n in names}})
        attempt['state'] = 'completed'
    except Exception as exc:
        attempt.update(state='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        attempt['seconds'] = time.time() - attempt['started_unix']
        save(attempt_path, attempt)


def released_official(bundle, record):
    from scib_benchmark.audit import audit_official_scib_score
    directory = f"metric_provenance/{record['run_id']}"
    raw = bundle.file(directory + '/raw.json')
    spec = load(bundle.file(directory + '/job.json'))
    backend = load(bundle.file(directory + '/backend.json'))
    errors = audit_official_scib_score(spec, raw)
    if errors:
        raise ValueError(str(errors))
    if any(spec.get(k) != v for k, v in {'dataset_id': record['dataset_id'],
        'dataset_sha256': record['dataset_sha256'], 'variant': record['method'], 'seed': int(record['seed'])}.items()):
        raise ValueError('Released official score identity differs from evaluation record')
    return load(raw), {'scope': 'authenticated_released_record_reuse', 'raw_path': str(raw),
        'publication_sha256': sha(raw), 'original_backend_record': backend,
        'original_vs_published_hashes': 'publication_provenance.json; original byte hashes are not hashes of path-normalized JSON'}


def score(config, bundle, output, args):
    import pandas as pd
    import score_experiment_v1_2 as metrics
    from scib_benchmark.scib_scoring import verify_paper_environment, PAPER_BACKEND
    records = bundle.records(config)
    new_ids = {r['run_id'] for r in training_records(bundle, records)}
    selected = [r for r in records if args.run_id is None or r['run_id'] == args.run_id]
    if not selected:
        raise ValueError('Unknown --run-id')
    for index, r in enumerate(selected):
        if sha(r['dataset_path']) != r['dataset_sha256']:
            raise ValueError('Raw dataset checksum differs from release')
        directory = output / 'scores' / r['run_id']
        if directory.exists():
            raise FileExistsError(f'Preserve existing score attempt and use a new output_root: {directory}')
        directory.mkdir(parents=True)
        local = config['mode'] == 'local' and r['run_id'] in new_ids
        if local:
            contract = load(Path(r['output_dir']) / 'local_contract.json')
            if contract['prepared_sha256'] != sha(output / 'prepared.json'):
                raise ValueError('Local training belongs to a different prepared contract')
            for name, digest in contract['artifacts'].items():
                if sha(Path(r['output_dir']) / name) != digest:
                    raise ValueError('Local training artifact changed')
            r['embedding_path'] = str(Path(r['output_dir']) / 'embedding.npz')
        split, obs = metrics.read_split(r)
        embedding = metrics.read_embedding(r, split)
        rows, predictions = metrics.classify(r, split, embedding)
        dataset = r['dataset_id']
        source = bundle.model(f'evaluation_source_cache/{dataset}/source_pca.npz')
        cache_status = load(bundle.model(f'evaluation_source_cache/{dataset}/status.json'))
        if cache_status.get('dataset_sha256') != r['dataset_sha256'] or cache_status.get('pcr_source_cache', {}).get('n_components') != 50:
            raise ValueError('Evaluation source PCA cache identity differs')
        rows.extend(metrics.biological_metrics(r, split, obs, embedding, source, output))
        selections = {}
        for field, eligible_dataset in metrics.COVARIATES.items():
            if dataset == eligible_dataset:
                for population in ('reference', 'query', 'joint'):
                    ids, path = metrics.within_label_selection(r, split, obs, population, field, output)
                    selections[(population, field)] = ids
        rows.extend(metrics.covariate_metrics(r, split, obs, embedding, selections))
        if local:
            env_path = output / 'official_environment.json'
            if not env_path.exists():
                verify_paper_environment(ROOT, env_path)
            historical = output / 'backend_source'
            cache = historical / 'scib_scores/source_cache' / PAPER_BACKEND / f"{dataset}__{r['dataset_sha256'][:16]}"
            cache.mkdir(parents=True, exist_ok=True)
            for name in ('status.json', 'source_pca.npz', 'pcr_before.json', 'cell_cycle_scores.npz', 'cell_cycle_before.json'):
                metrics.freeze_copy(bundle.model(f'paper_source_cache/{dataset}/{name}'), cache / name)
            runtime = dict(config, historical_run_root=str(historical))
            r['legacy_scib_path'] = None
            inputs = {'dataset': r['dataset_sha256'], 'split': sha(r['split_path']),
                      'features': sha(r['features_path']), 'embedding': sha(r['embedding_path']),
                      'source_manifest': sha(ROOT / 'source_file_mapping.json'),
                      'config': sha(args.config)}
            official, backend = metrics.official_metrics(r, inputs, runtime, ROOT, ROOT, output, directory, index)
            rows.extend(official)
        else:
            raw, backend = released_official(bundle, r)
            for name in metrics.OFFICIAL_METRICS:
                item = raw['official_scib_metrics'][name]
                fallback = bool(item.get('details', {}).get('fallback'))
                reason = item.get('reason', '')
                if fallback:
                    reason += '; HISTORICAL NON-EQUIVALENT FALLBACK: ' + json.dumps(item['details'], sort_keys=True)
                for metric in (name, metrics.ALIASES[name]) if name in metrics.ALIASES else (name,):
                    row = metrics.metric_row(r, metric, 'joint', item['value'], reason, item['status'])
                    row.update(implementation=item.get('implementation', ''), comparison_eligible=not fallback)
                    rows.append(row)
        pd.DataFrame(rows, columns=metrics.COLUMNS).to_csv(directory / 'metrics.csv', index=False)
        predictions.to_csv(directory / 'predictions.csv.gz', index=False)
        save(directory / 'backend.json', backend)
        save(directory / 'complete.json', {'identity': 'new_local_scoring' if local else 'released_embedding_reassessment',
            'run_id': r['run_id'], 'bundle_manifest_sha256': bundle.manifest_sha256,
            'source_hashes': source_hashes(), 'embedding_sha256': sha(r['embedding_path']),
            'outputs': {name: sha(directory / name) for name in ('metrics.csv', 'predictions.csv.gz', 'backend.json')}})


def aggregate(config, bundle, output, args):
    import pandas as pd
    if config['mode'] == 'released':
        input_dir = bundle.root
    else:
        records = bundle.records(config)
        frames = []
        for record in records:
            directory = output / 'scores' / record['run_id']
            completed = load(directory / 'complete.json')
            if completed['bundle_manifest_sha256'] != bundle.manifest_sha256 or completed['source_hashes'] != source_hashes():
                raise ValueError('Score contract differs from current source/bundle')
            for name, digest in completed['outputs'].items():
                if sha(directory / name) != digest:
                    raise ValueError('Completed score output changed')
            frames.append(pd.read_csv(directory / 'metrics.csv'))
        pd.concat(frames, ignore_index=True).to_csv(output / 'metrics.csv', index=False)
        columns = ['run_id', 'method', 'dimension', 'dataset_id', 'heldout_batch', 'seed', 'stochastic']
        pd.DataFrame(records)[columns].to_csv(output / 'runs.csv', index=False)
        input_dir = output
    for kind in ('tables', 'figures'):
        execute([interpreter(config['analysis_python']), ROOT / f'scripts/build_experiment_v1_2_{kind}.py',
                 '--input-dir', input_dir, '--output-dir', output])
    save(output / 'aggregation.json', {'mode': config['mode'], 'bundle_manifest_sha256': bundle.manifest_sha256,
        'metrics_sha256': sha(input_dir / 'metrics.csv'), 'runs_sha256': sha(input_dir / 'runs.csv'),
        'files': {str(p.relative_to(output)): sha(p) for kind in ('tables', 'figures')
                  for p in sorted((output / kind).iterdir()) if p.is_file()}})


def zip_entries(target, entries):
    if target.exists():
        raise FileExistsError(target)
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        lines = []
        for name, path in sorted(entries.items()):
            member_path(name)
            archive.write(path, name)
            lines.append(f'{sha(path)}  {name}\n')
        archive.writestr('MANIFEST.sha256', ''.join(lines))
    target.with_suffix('.zip.sha256').write_text(f'{sha(target)}  {target.name}\n')
    return {'filename': target.name, 'sha256': sha(target), 'bytes': target.stat().st_size,
            'members': sorted(entries)}


def package(config, bundle, output, args):
    aggregation = load(output / 'aggregation.json')
    if aggregation['mode'] != config['mode'] or aggregation['bundle_manifest_sha256'] != bundle.manifest_sha256:
        raise ValueError('Aggregate this exact mode and bundle before packaging')
    input_dir = bundle.root if config['mode'] == 'released' else output
    # Carry only the authenticated, explicitly enumerated bundle members.
    entries = {name: bundle.file(name) for name in bundle.files}
    entries.update({name: input_dir / name for name in ('metrics.csv', 'runs.csv')})
    for name, digest in aggregation['files'].items():
        path = output / name
        if sha(path) != digest:
            raise ValueError('Aggregation output changed')
        entries[name] = path
    for name in ('metrics.csv', 'runs.csv'):
        if sha(entries[name]) != aggregation[name.removesuffix('.csv') + '_sha256']:
            raise ValueError('Aggregation inputs changed')
    for kind in ('tables', 'figures'):
        entries[f'scripts/build_experiment_v1_2_{kind}.py'] = ROOT / f'scripts/build_experiment_v1_2_{kind}.py'
    entries['aggregation.json'] = output / 'aggregation.json'
    entries['source_file_mapping.json'] = ROOT / 'source_file_mapping.json'
    entries['publication_provenance.json'] = bundle.file('publication_provenance.json')
    parts = load(bundle.file('model_artifacts.json'))
    if config['mode'] == 'local':
        prepared = load(output / 'prepared.json')
        model_entries, part_number, part_bytes = {}, 1, 0
        # Stay below the public 1.8-GiB ceiling even after ZIP headers.
        byte_limit = 1800 * 1024 * 1024
        def flush_models():
            nonlocal model_entries, part_number, part_bytes
            if model_entries:
                part = zip_entries(output / f'SOLAR_v1.2_Local_Models_part{part_number:02d}.zip', model_entries)
                if part['bytes'] > 1932735283:
                    raise ValueError('Local model attachment exceeds the release size ceiling')
                parts.append(part)
                model_entries, part_bytes = {}, 0
                part_number += 1
        for record in sorted(prepared['runs'], key=lambda r: r['run_id']):
            run_id = record['run_id']
            directory = Path(record['output_dir'])
            contract = load(directory / 'local_contract.json')
            if contract['prepared_sha256'] != sha(output / 'prepared.json'):
                raise ValueError('Local model contract belongs to another preparation')
            entries[f'local_training/{run_id}/local_contract.json'] = directory / 'local_contract.json'
            entries[f'local_training/{run_id}/attempt.json'] = output / 'attempts' / f'{run_id}.json'
            for name, digest in sorted(contract['artifacts'].items()):
                path = directory / name
                if sha(path) != digest:
                    raise ValueError('Local training artifact changed before package')
                if name.endswith(('.pt', '.npz')) or name == 'model/solar_metadata.json':
                    size = path.stat().st_size
                    if size > byte_limit:
                        raise ValueError('Single local artifact exceeds attachment ceiling')
                    if part_bytes + size > byte_limit:
                        flush_models()
                    model_entries[f'local_models/{run_id}/{name}'] = path
                    part_bytes += size
                else:
                    entries[f'local_training/{run_id}/{name}'] = path
        flush_models()
        entries['local_prepared.json'] = output / 'prepared.json'
        for record in bundle.records(config):
            directory = output / 'scores' / record['run_id']
            for name in ('metrics.csv', 'predictions.csv.gz', 'backend.json', 'complete.json'):
                entries[f'local_scores/{record["run_id"]}/{name}'] = directory / name
        save(output / 'model_artifacts.json', parts)
        entries['model_artifacts.json'] = output / 'model_artifacts.json'
    save(output / 'export_scope.json', {'identity': 'local_reproduction_export', 'mode': config['mode'],
        'bundle_manifest_sha256': bundle.manifest_sha256,
        'released_models': 'Original model_artifacts entries remain separate authenticated release attachments',
        'local_models': 'local_models/RUN_ID members in the additional Local_Models attachments',
        'original_evidence': 'Original provenance and runs retained; local_training/local_scores are distinct new evidence',
        'not_the_published_release': True})
    entries['export_scope.json'] = output / 'export_scope.json'
    target = output / 'SOLAR_v1.2_Local_Reproduction.zip'
    zip_entries(target, entries)
    print(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inventory', 'audit', 'prepare', 'run', 'score', 'aggregate', 'package'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--method', choices=('solar30', 'sclsc30'))
    parser.add_argument('--dataset')
    parser.add_argument('--batch')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--run-id', help='score only this explicitly registered record; omission scores all records')
    args = parser.parse_args()
    args.config = args.config.resolve()
    config = yaml.safe_load(args.config.read_text())
    if config.get('version') != 'v1.2' or config.get('mode') not in ('released', 'local'):
        raise ValueError('Public config requires version: v1.2 and mode: released|local')
    if Path(sys.prefix).resolve() != interpreter(config['analysis_python']).parent.parent.resolve():
        raise RuntimeError('Invoke this entry with the configured locked analysis interpreter')
    execute([interpreter(config['analysis_python']), ROOT / 'scripts/check_method_environment.py', '--method', 'analysis'])
    output = resolve(config['output_root'])
    bundle = Bundle(config)
    if output.is_relative_to(bundle.root) or output.is_relative_to(bundle.attachments):
        raise ValueError('Output must be separate from authenticated released artifacts')
    output.mkdir(parents=True, exist_ok=True)
    globals()[args.command](config, bundle, output, args)


if __name__ == '__main__':
    main()
