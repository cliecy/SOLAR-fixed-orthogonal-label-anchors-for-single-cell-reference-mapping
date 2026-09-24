import hashlib
from pathlib import Path
import subprocess
import sys
import venv
import zipfile

import pytest
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
try:
    from experiment_v1_2 import interpreter, training_predictions
    from materialize_experiment_v1_2 import extract_verified, sha
finally:
    sys.path.remove(str(SCRIPTS))


def test_interpreter_preserves_virtual_environment(tmp_path):
    environment = tmp_path / 'environment'
    venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    output = subprocess.check_output([str(interpreter(environment / 'bin/python')), '-c',
                                      'import sys; print(sys.prefix)'], text=True)
    assert Path(output.strip()).resolve() == environment.resolve()


@pytest.mark.parametrize('corruption', ['traversal', 'content'])
def test_archive_rejection_does_not_publish_partial_or_outside_files(tmp_path, corruption):
    archive = tmp_path / 'input.zip'
    name = '../outside.txt' if corruption == 'traversal' else 'result.txt'
    payload = b'actual bytes'
    declared = hashlib.sha256(b'other bytes' if corruption == 'content' else payload).hexdigest()
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr(name, payload)
        handle.writestr('MANIFEST.sha256', f'{declared}  {name}\n')
    target = tmp_path / 'extracted'
    with pytest.raises(ValueError):
        extract_verified(archive, sha(archive), target)
    assert not target.exists()
    assert not (tmp_path / 'outside.txt').exists()


def test_training_readout_uses_full_reference_and_distance_weights():
    reference = np.r_[np.linspace(1, 2, 20), 0.01][:, None]
    embedding = np.vstack([reference, [[0.0]]])
    result = training_predictions(embedding, np.array(['far'] * 20 + ['near']), np.array(['query-cell']))
    assert result.to_dict('records') == [{'barcode': 'query-cell', 'prediction': 'near'}]
