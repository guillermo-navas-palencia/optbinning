"""Checks for the comparison itself; never assert noisy timing thresholds."""
import copy

import pytest
from ortools.sat.python import cp_model

from bench_cp import summarize
from cases import canonical_hash


def pair(stage="construction", status="OPTIMAL"):
    row = dict(stage=stage, case=["example"], input_hash="same", seconds=2,
               model_hash="model", output_hash="prediction", status=status)
    left = dict(versions={"ortools": "same"}, rows=[row])
    right = copy.deepcopy(left)
    right["rows"][0]["seconds"] = 1
    return dict(reference=left, candidate=right)


def test_sign_normalization_keeps_actual_bound_changes():
    def model(reverse=False, bound=3):
        cp = cp_model.CpModel()
        x = cp.NewIntVar(0, 10, "x")
        cp.Add(-x >= -bound if reverse else x <= bound)
        return cp
    assert canonical_hash(model()) == canonical_hash(model(True))
    assert canonical_hash(model()) != canonical_hash(model(bound=4))


def test_paired_speedups():
    summary = summarize([pair(), pair()], True)[0]
    assert summary["eligible_pairs"] == 2
    assert summary["speedup_median"] == 2


@pytest.mark.parametrize("field", ["input_hash", "model_hash"])
def test_changed_inputs_or_models_rejected(field):
    data = pair()
    data["candidate"]["rows"][0][field] = "different"
    with pytest.raises(ValueError):
        summarize([data], True)


def test_incomplete_coverage_rejected():
    data = pair()
    data["candidate"]["rows"] = []
    with pytest.raises(ValueError):
        summarize([data], True)


@pytest.mark.parametrize("status", ["FEASIBLE", "UNKNOWN", "INFEASIBLE"])
def test_nonoptimal_fit_not_ranked(status):
    summary = summarize([pair("fit", status)], True)[0]
    assert summary["eligible_pairs"] == 0
    assert summary["speedup_median"] is None


def test_changed_model_retained_without_speedup():
    data = pair()
    data["candidate"]["rows"][0]["model_hash"] = "changed"
    summary = summarize([data], False)[0]
    assert not summary["equivalent"]
    assert summary["speedup_median"] is None


def test_worker_failure_retains_completed_side(tmp_path, monkeypatch):
    import json
    import subprocess
    import sys
    import bench_cp
    from types import SimpleNamespace
    monkeypatch.setattr(bench_cp, 'git', lambda path, *args: '' if args[0] == 'status' else 'fixture')
    output = tmp_path / 'pair.json'
    monkeypatch.setattr(sys, 'argv', ['bench', '--reference', str(tmp_path),
        '--candidate', str(tmp_path), '--output', str(output), '--repeats', '1'])
    calls = []
    def worker(*args, **kwargs):
        calls.append(args)
        if len(calls) == 2:raise subprocess.CalledProcessError(1, args[0])
        return SimpleNamespace(stdout=json.dumps(pair()['reference']), stderr='')
    monkeypatch.setattr(bench_cp.subprocess, 'run', worker)
    with pytest.raises(subprocess.CalledProcessError):bench_cp.main()
    saved = json.loads(output.read_text())
    assert not saved['complete']
    assert len(saved['pairs']) == 1
    first, second = saved['pairs'][0]['order']
    assert saved['pairs'][0][first]['rows']
    assert second not in saved['pairs'][0]
