"""Solver-only DP/CP comparison; experimental DP guards disabled in memory."""
import argparse
import inspect
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import ortools

import optbinning.binning.dp as dp_module
from optbinning.binning.cp import BinningCP


def unguarded_dp():
    # Keep production eligibility unchanged. Fail if its implementation changes.
    source = inspect.getsource(dp_module.solve_iv)
    replacements = {
        'n > 100 or ': '',
        'started + min(.1, time_limit / 10)': "float('inf')",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError('DP guards changed; review this benchmark.')
        source = source.replace(old, new)
    namespace = dict(dp_module.__dict__)
    exec(compile(source, '<benchmark-only DP>', 'exec'), namespace)
    return namespace['solve_iv']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prebins', type=int, nargs='+',
                        default=[20, 50, 100, 150, 200, 300])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--cp-time-limit', type=float, default=2)
    parser.add_argument('--output', type=Path,
                        default=Path('benchmark-artifacts/dp_bin_limits'))
    args = parser.parse_args()
    if (args.repetitions < 1 or args.cp_time_limit <= 0
            or any(n < 6 or n > 1000 for n in args.prebins)
            or len(set(args.prebins)) != len(args.prebins)):
        parser.error('Use unique prebin counts in [6, 1000] and positive budgets.')
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists()
           for name in ['results.json', 'manifest.json', 'summary.txt']):
        parser.error('Choose a fresh output directory.')
    solve_dp = unguarded_dp()
    manifest = dict(
        platform=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, ortools=ortools.__version__,
        revision=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        seed='73135 + prebin count', prebins=args.prebins,
        repetitions=args.repetitions, cp_search_seconds=args.cp_time_limit,
        cp_workers=1, dp_guards_disabled=['100-prebin cap', 'attempt budget'],
        note='Synthetic positive integer counts; includes model construction; '
             'excludes imports/prebinning. No timing assertions. Unlimited '
             'bins is an easy control: retaining all prebins is IV-optimal.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    summary = ['prebins | max_bins | DP seconds | CP seconds | CP statuses']
    for n in args.prebins:
        rng = np.random.default_rng(73135 + n)
        event = rng.integers(1, 100, n)
        nonevent = rng.integers(1, 100, n)
        for max_bins in [6, None]:
            for rep in range(args.repetitions):
                pair = {}
                for method in (['dp', 'cp'] if rep % 2 == 0 else ['cp', 'dp']):
                    started = time.perf_counter()
                    build = None
                    if method == 'dp':
                        status, solution, objective, _ = solve_dp(
                            nonevent, event, None, max_bins, None, None, 2)
                        bound = objective
                    else:
                        model = BinningCP(
                            None, None, max_bins, None, None, None, None,
                            None, None, 0, None, 'consecutive', 0, None,
                            args.cp_time_limit)
                        model.build_model('iv', nonevent, event, None)
                        build = time.perf_counter() - started
                        status, solution = model.solve()
                        objective = (model.solver_.ObjectiveValue()
                                     if status in ('OPTIMAL', 'FEASIBLE')
                                     else None)
                        bound = model.solver_.BestObjectiveBound()
                    row = dict(prebins=n, max_bins=max_bins, rep=rep,
                               method=method, seconds=time.perf_counter()-started,
                               build_seconds=build, status=status,
                               objective=objective, bound=bound,
                               bins=(int(solution.sum()) if status in
                                     ('OPTIMAL', 'FEASIBLE') else None))
                    rows.append(row)
                    pair[method] = row
                if pair['dp']['status'] != 'OPTIMAL':
                    raise RuntimeError('Unguarded DP did not prove optimality.')
                best = pair['dp']['objective']
                cp = pair['cp']
                if cp['status'] == 'OPTIMAL' and cp['objective'] != best:
                    raise RuntimeError('Proven optimal objectives disagree.')
                if cp['status'] == 'FEASIBLE' and not (
                        cp['objective'] <= best + 1e-6 <= cp['bound'] + 1e-6):
                    raise RuntimeError('DP optimum outside CP bounds.')
                (args.output / 'results.json').write_text(
                    json.dumps(rows, indent=2))
            case = [r for r in rows if r['prebins'] == n
                    and r['max_bins'] == max_bins]
            timings = [statistics.median(r['seconds'] for r in case
                                        if r['method'] == method)
                       for method in ['dp', 'cp']]
            statuses = [r['status'] for r in case if r['method'] == 'cp']
            line = (f'{n} | {max_bins} | {timings[0]:.4f} | '
                    f'{timings[1]:.4f} | {statuses}')
            summary.append(line)
            print(line, flush=True)
            (args.output / 'summary.txt').write_text('\n'.join(summary) + '\n')


if __name__ == '__main__':
    main()
