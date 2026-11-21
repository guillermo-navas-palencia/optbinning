"""Publication rules: changed/partial cases cannot become performance claims."""
import copy

import pytest

from bench_cp import summarize
from test_bench_cp import pair
from track import panels, percentile, render


def report(pairs):
    return dict(complete=True, repeats=len(pairs), summary=summarize(pairs, False))


def test_changed_in_one_repeat_excludes_whole_case():
    changed = pair()
    changed['candidate']['rows'][0]['model_hash'] = 'different'
    row = panels(report([pair(), changed]))[0]
    assert row['speedup'] is None
    assert row['eligible'] == 0
    assert row['changed'] == 1
    assert row['reference_ms'] == 2000
    assert row['candidate_ms'] == 1000


def test_nonoptimal_in_one_repeat_excludes_whole_case():
    pairs = [pair('fit'), pair('fit', 'FEASIBLE')]
    for item in pairs:
        for side in ['reference', 'candidate']:
            item[side]['rows'][0]['case'] = ['binary', 'numerical', None]
    row = panels(report(pairs))[0]
    assert row['speedup'] is None
    assert row['changed'] == 0


def test_case_medians_not_repeat_pooling():
    first = pair()
    second = copy.deepcopy(first)
    second['candidate']['rows'][0]['seconds'] = .5
    row = panels(report([first, second]))[0]
    assert row['speedup'] == 3
    assert row['p5'] == row['p95'] == 3
    assert row['eligible'] == row['cases'] == 1


def test_percentiles_interpolate_observed_cases():
    assert percentile([1, 2, 3], .05) == pytest.approx(1.1)
    assert percentile([1, 2, 3], .95) == pytest.approx(2.9)


def test_partial_report_never_published():
    data = report([pair()])
    data['complete'] = False
    with pytest.raises(ValueError, match='Incomplete'):
        panels(data)


def test_render_preserves_versions_and_evidence_links(tmp_path):
    record = dict(label='Test update', revision='a' * 40, protocol='b' * 64,
                  measured_at='2026-11-10', runner='test', run_url='',
                  comparisons=[dict(label='1.0.0', reference='c' * 40,
                                    panels=panels(report([pair()])))])
    render(dict(records=[record]), tmp_path)
    rst = (tmp_path / 'index.rst').read_text()
    assert '2.00×' in rst
    assert '271c906d5250f17633396f88d246d64b4c449380' in rst
    assert 'data/' + 'a' * 40 + '.json.gz' in rst
    assert 'Changed' in (tmp_path / 'index.html').read_text()


def test_history_adds_code_updates_but_not_documentation_duplicates(tmp_path, monkeypatch):
    import json
    import sys
    import track
    root = tmp_path / 'harness'
    root.mkdir()
    for name in ['bench_cp.py', 'cases.py', 'track.py', 'requirements.txt']:
        (root / name).write_text('fixture')
    (root / 'updates.json').write_text(json.dumps(dict(
        baseline='base', baseline_label='1.0.0',
        updates=[dict(revision='first', label='First change')])) )
    trees = dict(base='tree0', first='tree1', newer='tree2', doc='tree2',
                 branch='tree3')
    monkeypatch.setattr(track, 'HERE', root)
    monkeypatch.setattr(track.platform, 'platform', lambda: 'test-runner')
    def fake_git(repo, command, ref):
        return trees[ref[:-11]] if ref.endswith(':optbinning') else ref.removesuffix('^{commit}')
    monkeypatch.setattr(track, 'git', fake_git)
    measured = []
    def fake_run(command, **kwargs):
        if '--output' in command:
            measured.append(command)
            Path = type(root)
            Path(command[command.index('--output') + 1]).write_text(json.dumps(report([pair()])))
    monkeypatch.setattr(track.subprocess, 'run', fake_run)
    out = tmp_path / 'results'
    def run(candidate, previous, refresh=False):
        monkeypatch.setattr(sys, 'argv', ['track', '--repository', str(tmp_path),
            '--output', str(out), '--candidate', candidate, '--previous', previous]
            + (['--refresh'] if refresh else []))
        track.main()
        return json.loads((out / 'history.json').read_text())
    assert len(run('first', 'base')['records']) == 1
    assert len(measured) == 1
    history = run('newer', 'first')
    assert len(history['records']) == 2
    assert len(measured) == 3  # baseline plus the preceding update
    assert history['records'][-1]['comparisons'][1]['reference'] == 'first'
    assert len(run('doc', 'newer')['records']) == 2
    assert len(measured) == 3
    assert len(list((out / 'data').glob('*.gz'))) == 2

    # Refresh from a documentation-only head must include automatically
    # registered updates, keeping their originally paired predecessor.
    history = run('doc', 'newer', refresh=True)
    assert len(history['records']) == 2
    assert len(measured) == 6
    assert history['records'][-1]['comparisons'][1]['reference'] == 'first'
    assert len(list((out / 'data').glob('*.gz'))) == 2

    # A protocol change must also refresh all retained code updates.
    (root / 'cases.py').write_text('updated fixture')
    history = run('doc', 'newer')
    assert len(measured) == 9
    assert len({r['protocol'] for r in history['records']}) == 1
    assert [r['revision'] for r in history['records']] == ['first', 'newer']

    # Different branches can have different predecessors. Refresh must not
    # silently turn their registration order into the comparison baseline.
    run('branch', 'first')
    history = run('doc', 'newer', refresh=True)
    branch = next(r for r in history['records'] if r['revision'] == 'branch')
    assert branch['comparisons'][1]['reference'] == 'first'
    assert len(measured) == 16

    # A refresh failure must preserve the accepted history/raw records while
    # retaining the new measurements outside disposable worktrees.
    import gzip
    import subprocess
    old_history = (out / 'history.json').read_bytes()
    old_raw = {p.name: p.read_bytes() for p in (out / 'data').glob('*.gz')}
    for mode in ['worker', 'validation']:
        def failed_run(command, **kwargs):
            if '--output' in command:
                partial = report([pair()])
                partial['complete'] = False
                destination = type(root)(command[command.index('--output') + 1])
                destination.write_text(json.dumps(partial))
                if mode == 'worker':
                    raise subprocess.CalledProcessError(1, command)
        monkeypatch.setattr(track.subprocess, 'run', failed_run)
        with pytest.raises((subprocess.CalledProcessError, ValueError)):
            run('doc', 'newer', refresh=True)
        pending = json.loads(gzip.decompress((out / 'pending.json.gz').read_bytes()))
        assert not pending['complete']
        assert len(pending['reports']) == 1
        assert pending['reports'][0]['summary']
        assert pending['error']
        assert (out / 'history.json').read_bytes() == old_history
        assert {p.name: p.read_bytes() for p in (out / 'data').glob('*.gz')} == old_raw
        assert not (out / 'pending-pair.json').exists()
    monkeypatch.setattr(track.subprocess, 'run', fake_run)
    run('doc', 'newer', refresh=True)
    assert not (out / 'pending.json.gz').exists()
