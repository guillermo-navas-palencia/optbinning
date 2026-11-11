"""Maintain one reproducible benchmark record per code revision, not run archives."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile

from bench_cp import git

HERE = Path(__file__).resolve().parent


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def panels(report):
    """Summarize within cases first; never pool construction and fit timings."""
    if not report.get('complete'):
        raise ValueError('Incomplete comparisons cannot be published')
    groups = {}
    for row in report['summary']:
        case = row['case']
        name = f"{row['stage']} / {case[0]}"
        if row['stage'] == 'fit':
            name += f' / {case[1]}'
        groups.setdefault(name, []).append(row)
    result = []
    for name, rows in groups.items():
        eligible = [r for r in rows if r['equivalent'] and
                    r['eligible_pairs'] == report['repeats']]
        speeds = [r['speedup_median'] for r in eligible]
        result.append(dict(
            panel=name, cases=len(rows), eligible=len(eligible),
            changed=sum(not r['equivalent'] for r in rows),
            reference_ms=1000 * statistics.median(
                statistics.median(r['reference_seconds']) for r in rows),
            candidate_ms=1000 * statistics.median(
                statistics.median(r['candidate_seconds']) for r in rows),
            speedup=statistics.median(speeds) if speeds else None,
            p5=percentile(speeds, .05) if speeds else None,
            p95=percentile(speeds, .95) if speeds else None))
    return result


def rst_table(headers, rows):
    lines = ['.. list-table::', '   :header-rows: 1', '']
    for row in [headers, *rows]:
        lines.append('   * - ' + str(row[0]))
        lines.extend('     - ' + str(value) for value in row[1:])
    return '\n'.join(lines) + '\n\n'


def save_pending(path, record):
    """Keep one recoverable attempt, separate from accepted revision records."""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(gzip.compress(json.dumps(record).encode(), mtime=0))
    temporary.replace(path)


def render(history, out):
    text = '''CP benchmark updates
====================

The fixed reference is OptBinning **1.0.0**, commit
``271c906d5250f17633396f88d246d64b4c449380``. Every comparison reruns both
revisions together on the same runner. These are CP construction and small
complete-fit controls, not predictive-performance measurements.

Construction uses 112 seeded synthetic aggregate cases. Complete fitting uses
8 controls from scikit-learn's bundled breast-cancer and diabetes datasets,
using their first feature (numerical or quantile-derived categorical), with
no shape constraint or native automatic shape selection. No data download is needed.

Speedup = reference time / candidate time; above 1 is faster. Each case first
uses the median of 7 paired repetitions. Panel speedups and p5–p95 summarize
those case medians, only for cases equivalent and eligible in **all** repeats.
Percentiles describe case variation, not confidence intervals. Timings are
median case times over all cases, including changed cases; their quotient
need not equal the restricted speedup. Changed models receive no equivalent-work
speedup. Fit eligibility requires OPTIMAL status and identical predictions.
A changed model is not itself a failure: correctness fixes need regression tests.

Runs from different dates or protocols are not a controlled head-to-head
comparison. Use the measured preceding-update comparison for incremental gains.
Hosted runners are noisy; no timing threshold gates a merge.

One compact record and compressed raw measurement file per measured revision
are retained here. Rerunning that revision replaces its record. This is not a
collection of old notebook packages. Every code update on an enabled branch
is added automatically; documentation-only changes reuse the last code record.

'''
    for record in history['records']:
        title = record['label'] + ' (' + record['revision'][:7] + ')'
        text += title + '\n' + '-' * len(title) + '\n\n'
        text += f"Measured {record['measured_at']}. Protocol ``{record['protocol'][:12]}``. "
        text += f"Runner: {record['runner']}.\n\n"
        text += f"`Raw paired measurements <data/{record['revision']}.json.gz>`__. "
        text += f"`Workflow run <{record['run_url']}>`__.\n\n" if record['run_url'] else '\n\n'
        for comparison in record['comparisons']:
            text += f"**Against {comparison['label']} ({comparison['reference'][:7]})**\n\n"
            rows = []
            for p in comparison['panels']:
                speed = '—' if p['speedup'] is None else f"{p['speedup']:.2f}× ({p['p5']:.2f}–{p['p95']:.2f})"
                rows.append([p['panel'], f"{p['reference_ms']:.3f}",
                             f"{p['candidate_ms']:.3f}", speed,
                             f"{p['eligible']}/{p['cases']}", p['changed']])
            text += rst_table(['Stage / cases', 'Reference ms', 'Update ms',
                               'Speedup (p5–p95)', 'Eligible', 'Changed'], rows)
    (out / 'index.rst').write_text(text)
    from docutils.core import publish_string
    (out / 'index.html').write_bytes(publish_string(
        text, writer_name='html5', settings_overrides={'halt_level': 2}))
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as stream:
            stream.write('## CP benchmarks against 1.0.0\n\n')
            stream.write(f"{len(history['records'])} measured revisions. Download the "
                         "benchmark-documentation artifact for tables and raw measurements.\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', default='HEAD')
    parser.add_argument('--previous', help='Previous push SHA or PR base SHA')
    parser.add_argument('--label', default='Current code update')
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    repo, out = args.repository.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    config = json.loads((HERE / 'updates.json').read_text())
    baseline = git(repo, 'rev-parse', config['baseline'] + '^{commit}')
    candidate = git(repo, 'rev-parse', args.candidate + '^{commit}')
    points = [{**p, 'revision': git(repo, 'rev-parse', p['revision'] + '^{commit}')}
              for p in config['updates']]
    history_path = out / 'history.json'
    history = json.loads(history_path.read_text()) if history_path.exists() else dict(records=[])
    # Automatically discovered updates are registered too. Include them when
    # refreshing or changing protocol, even from a documentation-only head.
    registered = {p['revision'] for p in points}
    for record in history['records']:
        if record['revision'] not in registered:
            points.append(dict(revision=record['revision'], label=record['label']))
            registered.add(record['revision'])
    # A documentation/harness-only commit does not become a fake library update.
    code_tree = lambda revision: git(repo, 'rev-parse', revision + ':optbinning')
    protocol = hashlib.sha256(b''.join((HERE / name).read_bytes() for name in
        ['bench_cp.py', 'cases.py', 'track.py', 'requirements.txt']) +
        platform.python_version().encode()).hexdigest()
    known_trees = {code_tree(p['revision']) for p in points}
    known_trees.update(r.get('code_tree') for r in history['records']
                       if r['protocol'] == protocol)
    if code_tree(candidate) not in known_trees:
        points.append(dict(revision=candidate, label=args.label))
    if history.get('baseline', baseline) != baseline:
        raise ValueError('A new baseline requires a separate benchmark series')
    history['baseline'] = baseline
    pending_path = out / 'pending.json.gz'
    previous = baseline
    with tempfile.TemporaryDirectory(prefix='cp-bench-') as temporary:
        checkouts = {}
        def checkout(revision):
            if revision not in checkouts:
                path = Path(temporary) / revision
                subprocess.run(['git', '-C', str(repo), 'worktree', 'add',
                                '--detach', str(path), revision], check=True)
                checkouts[revision] = path
            return checkouts[revision]
        try:
            for point in points:
                revision = point['revision']
                if revision == candidate and args.previous and set(args.previous) != {'0'}:
                    previous = git(repo, 'rev-parse', args.previous + '^{commit}')
                old = next((r for r in history['records'] if r['revision'] == revision), None)
                if old and old['protocol'] == protocol and not args.refresh:
                    previous = revision
                    continue
                refs = [(baseline, config['baseline_label'])]
                if old:
                    # Refresh the same comparison, not whichever branch
                    # happened to be measured immediately before this one.
                    refs = [(c['reference'], c['label']) for c in old['comparisons']]
                elif previous != baseline and previous != revision:
                    refs.append((previous, 'preceding update'))
                comparisons, raw = [], []
                for reference, label in refs:
                    # Directly recoverable even if the parent process is killed.
                    output = out / 'pending-pair.json'
                    output.unlink(missing_ok=True)
                    attempt = dict(complete=False, revision=revision, reference=reference,
                        protocol=protocol, run_url=os.environ.get('BENCHMARK_RUN_URL', ''),
                        reports=list(raw))
                    save_pending(pending_path, attempt)
                    try:
                        subprocess.run([sys.executable, str(HERE / 'bench_cp.py'),
                            '--reference', str(checkout(reference)), '--candidate',
                            str(checkout(revision)), '--repeats', '7', '--output', str(output)], check=True)
                        report = json.loads(output.read_text())
                        raw.append(report)
                        attempt['reports'] = list(raw)
                        save_pending(pending_path, attempt)  # Before validation/rendering.
                        comparisons.append(dict(label=label, reference=reference, panels=panels(report)))
                        output.unlink(missing_ok=True)
                    except Exception as error:
                        attempt['error'] = repr(error)
                        if output.exists() and len(attempt['reports']) == len(comparisons):
                            try:attempt['reports'].append(json.loads(output.read_text()))
                            except (OSError, ValueError) as read_error:attempt['read_error'] = repr(read_error)
                        save_pending(pending_path, attempt)
                        output.unlink(missing_ok=True)
                        raise
                record = dict(**point, protocol=protocol, code_tree=code_tree(revision),
                    comparisons=comparisons,
                    measured_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                    runner=platform.platform(), run_url=os.environ.get('BENCHMARK_RUN_URL', ''))
                raw_path = out / 'data' / f'{revision}.json.gz'
                raw_path.write_bytes(gzip.compress(json.dumps(raw).encode(), mtime=0))
                history['records'] = [r for r in history['records'] if r['revision'] != revision]
                history['records'].append(record)
                history_path.write_text(json.dumps(history, indent=2) + '\n')
                render(history, out)
                previous = revision
        finally:
            for path in checkouts.values():
                subprocess.run(['git', '-C', str(repo), 'worktree', 'remove', '--force', str(path)], check=True)
    render(history, out)
    pending_path.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
