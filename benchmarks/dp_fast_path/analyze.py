"""Validate benchmark records and report paired timing distributions."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmark-artifacts/dp_fast_path'))
    args = parser.parse_args()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    rows = json.loads((args.output / 'results.json').read_text())
    expected = manifest['expected_pairs']
    assert len(rows) == expected * 2, 'Incomplete benchmark'
    assert len({(r['block'], r['method']) for r in rows}) == len(rows), 'Duplicate records'
    pairs = []
    for block in range(expected):
        pair = {r['method']: r for r in rows if r['block'] == block}
        assert set(pair) == {'cp', 'auto'}, 'Missing method'
        cp, fast = pair['cp'], pair['auto']
        for field in ['dataset', 'feature', 'dtype', 'prebins', 'rep']:
            assert cp[field] == fast[field], 'Unpaired configurations'
        for row in pair.values():
            assert np.isfinite(row['fit_seconds']) and row['fit_seconds'] > 0
            assert row['backend'] in {'cp', 'dp'}
        if cp['status'] == fast['status'] == 'OPTIMAL':
            assert cp['objective'] is not None and cp['objective'] == fast['objective'], pair
        elif fast['status'] == 'OPTIMAL' and cp['status'] == 'FEASIBLE':
            assert cp['objective'] <= fast['objective'] <= cp['bounds'], pair
        if manifest['mode'] == 'smoke':
            assert cp['status'] == fast['status'] == 'OPTIMAL', pair
            if cp['dtype'] == 'categorical':
                assert fast['backend'] == 'cp', pair
        pairs.append((cp, fast))
    summary = []
    groups = sorted({(r['dataset'], r['dtype'], r['prebins']) for r in rows})
    for dataset, dtype, prebins in groups:
        group = [(a, b) for a, b in pairs if (a['dataset'], a['dtype'], a['prebins']) == (dataset, dtype, prebins)]
        both = [(a, b) for a, b in group if a['status'] == b['status'] == 'OPTIMAL']
        ratios = [a['fit_seconds'] / b['fit_seconds'] for a, b in both]
        summary.append(dict(dataset=dataset, dtype=dtype, prebins=prebins,
            pairs=len(group), paired_optima=len(both),
            cp_status=dict(Counter(a['status'] for a, b in group)),
            auto_status=dict(Counter(b['status'] for a, b in group)),
            auto_backend=dict(Counter(b['backend'] for a, b in group)),
            paired_speedup_quantiles=dict(zip(['min', 'p25', 'median', 'p75', 'max'],
                map(float, np.quantile(ratios, [0, .25, .5, .75, 1])))) if ratios else None))
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
