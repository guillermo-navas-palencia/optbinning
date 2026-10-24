"""Paired public-estimator benchmark; no network access or timing gate."""
import argparse
import hashlib
import json
import platform
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np
import ortools
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer

from optbinning import OptimalBinning


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('benchmark-artifacts/dp_fast_path'))
    parser.add_argument('--telco', type=Path, help='Optional existing IBM Telco CSV; never downloaded automatically')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ['results.json', 'manifest.json']):
        parser.error('Choose a fresh output directory; existing evidence is not overwritten.')
    cancer = load_breast_cancer()
    columns = [0] if args.smoke else range(cancer.data.shape[1])
    cases = [('breast_cancer', str(cancer.feature_names[j]), 'numerical',
              cancer.data[:, j], cancer.target) for j in columns]
    # A deterministic, mixed-class categorical control exercises CP fallback.
    cases.append(('synthetic_control', 'category', 'categorical',
                  np.repeat(['a', 'b', 'c'], 100),
                  np.concatenate([np.r_[np.zeros(100-n), np.ones(n)] for n in [20, 50, 80]])))
    telco_metadata = None
    if args.telco:
        telco = pd.read_csv(args.telco)
        for name in ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'PaymentMethod']:
            dtype = 'numerical' if name in ['tenure', 'MonthlyCharges', 'TotalCharges'] else 'categorical'
            x = (pd.to_numeric(telco[name], errors='coerce').to_numpy()
                 if dtype == 'numerical' else telco[name].to_numpy())
            cases.append(('telco', name, dtype, x, (telco.Churn == 'Yes').astype(int).to_numpy()))
        telco_metadata = {'rows': len(telco), 'sha256': hashlib.sha256(args.telco.read_bytes()).hexdigest()}
    budgets, repetitions = ([20], 1) if args.smoke else ([20, 40], 3)
    blocks = [(i, m, rep) for i in range(len(cases)) for m in budgets for rep in range(repetitions)]
    np.random.default_rng(73135).shuffle(blocks)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    manifest = dict(mode='smoke' if args.smoke else 'full', expected_pairs=len(blocks),
                    python=platform.python_version(), platform=platform.platform(),
                    numpy=np.__version__, sklearn=sklearn.__version__, ortools=ortools.__version__,
                    revision=revision.stdout.strip() if revision.returncode == 0 else None,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    telco=telco_metadata, seed=73135, repetitions=repetitions,
                    note='Paired CP/auto on same runner; alternating order. Imports excluded. No timing threshold.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows = []
    for block, (index, m, rep) in enumerate(blocks):
        dataset, name, dtype, x, y = cases[index]
        for method in (['cp', 'auto'] if block % 2 == 0 else ['auto', 'cp']):
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                model = OptimalBinning(name=name, dtype=dtype, solver=method,
                    monotonic_trend=None, max_n_prebins=m, min_prebin_size=.01,
                    min_n_bins=3, max_n_bins=6, min_bin_size=.05, time_limit=2).fit(x, y)
            rows.append(dict(block=block, dataset=dataset, feature=name, dtype=dtype,
                prebins=m, rep=rep, method=method, backend=model.solver_backend_,
                fit_seconds=time.perf_counter() - started, status=model.status,
                objective=model._optimizer.get('objective') if model._optimizer else None,
                bounds=model._optimizer.get('best_objective_bound') if model._optimizer else None,
                bins=len(model.splits) + 1,
                warnings=sorted(set(str(w.message) for w in caught))))
            (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
        print(f'{block + 1}/{len(blocks)} paired fits', flush=True)


if __name__ == '__main__':
    main()
