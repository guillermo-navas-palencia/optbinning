"""Small counterexamples to invalid binning model constraints."""
import numpy as np
import pytest

from optbinning.binning.continuous_cp import ContinuousBinningCP
from optbinning.binning.cp import BinningCP
from optbinning.binning.mip import BinningMIP


@pytest.mark.parametrize("solver", ["cp", "mip"])
@pytest.mark.parametrize("trend,turn", [
    ("ascending", None), ("descending", None),
    ("peak_heuristic", 0), ("valley_heuristic", 6)])
def test_monotonic_pruning_preserves_valid_partition(solver, trend, turn):
    ne = np.array([3, 11, 16, 4, 10, 3, 6])
    e = np.array([18, 4, 16, 6, 17, 15, 1])
    if trend == "ascending":
        ne, e = ne[::-1], e[::-1]
    args = (trend, 1, 4, 10, 100, 3, 60, 3, 60, 0,
            None, "consecutive", 0, None)
    model = (BinningCP(*args, 5) if solver == "cp" else
             BinningMIP(*args, "cbc", 5))
    model.build_model("iv", ne, e, turn)
    status, cuts = model.solve()
    assert status == "OPTIMAL"
    # These feasible boundaries were excluded by monotonic preprocessing.
    expected = ([5, 7] if trend == "ascending" else
                [1, 2, 7] if trend == "valley_heuristic" else [2, 7])
    assert (np.flatnonzero(cuts) + 1).tolist() == expected
    groups = np.split(np.arange(7), np.flatnonzero(cuts)[:-1] + 1)
    rates = []
    for ids in groups:
        a, b = ne[ids].sum(), e[ids].sum()
        assert 3 <= a <= 60 and 3 <= b <= 60 and 10 <= a + b <= 100
        rates.append(b / (a + b))
    differences = np.diff(rates)
    if trend == "ascending":
        assert (differences >= 0).all()
    elif trend == "valley_heuristic":
        assert differences[0] <= 0 <= differences[1]
    else:
        assert (differences <= 0).all()


@pytest.mark.parametrize('trend,means', [
    ('peak', [-3, 2, 1]), ('valley', [3, -2, -1])])
@pytest.mark.parametrize('shift', [0, -10, 10])
def test_continuous_turning_shape_signed_means(trend, means, shift):
    n = np.array([10, 10, 10])
    means = np.array(means) + shift
    cp = ContinuousBinningCP(trend, 3, 3, None, None, 0, None,
                             'consecutive', 0, None, 5)
    cp.build_model(n, n * means, n * (means ** 2 + 1), None)
    status, cuts = cp.solve()
    # With exactly three bins there is only one possible partition; its
    # means have the requested shape, including across zero.
    assert status == 'OPTIMAL'
    assert cuts.all()


def _enumerated_monotonic_partitions(ne, e, trend, scaled):
    """Independent finite-grid oracle, using raw counts rather than model data."""
    feasible = {}
    for mask in range(1 << (len(e) - 1)):
        ends = [i + 1 for i in range(len(e) - 1) if mask & (1 << i)]
        ends.append(len(e))
        if len(ends) > 4:
            continue
        start = 0
        rates, objective = [], 0
        for end in ends:
            a, b = int(ne[start:end].sum()), int(e[start:end].sum())
            if not (3 <= a <= 60 and 3 <= b <= 60 and 10 <= a + b <= 100):
                break
            rate = b / (a + b)
            p, q = b / e.sum(), a / ne.sum()
            value = (p - q) * np.log(p / q)
            rates.append(int(1e6 * rate) if scaled else rate)
            objective += int(1e6 * value) if scaled else value
            start = end
        else:
            differences = np.diff(rates)
            valid = (np.all(differences >= 0) if trend == 'ascending'
                     else np.all(differences <= 0))
            if valid:
                feasible[tuple(ends)] = objective
    return feasible


@pytest.mark.parametrize('solver', ['cp', 'mip'])
@pytest.mark.parametrize('trend', ['ascending', 'descending'])
@pytest.mark.parametrize('case', ['counterexample', 'ties', 0, 1, 2, 3])
def test_monotonic_optimum_matches_exhaustive_partitions(solver, trend, case):
    if case == 'counterexample':
        ne = np.array([3, 11, 16, 4, 10, 3, 6])
        e = np.array([18, 4, 16, 6, 17, 15, 1])
        if trend == 'ascending':
            ne, e = ne[::-1], e[::-1]
    elif case == 'ties':
        ne = e = np.full(5, 5)
    else:
        rng = np.random.default_rng(case)
        ne, e = rng.integers(1, 20, size=(2, 5))
    feasible = _enumerated_monotonic_partitions(ne, e, trend, solver == 'cp')
    args = (trend, 1, 4, 10, 100, 3, 60, 3, 60, 0,
            None, 'consecutive', 0, None)
    model = (BinningCP(*args, 5) if solver == 'cp'
             else BinningMIP(*args, 'cbc', 5))
    model.build_model('iv', ne, e, None)
    status, cuts = model.solve()
    if not feasible:
        assert status == 'INFEASIBLE'
        return
    assert status == 'OPTIMAL'
    selected = tuple(np.flatnonzero(cuts) + 1)
    assert selected in feasible
    # Equal optima may have different cuts. CP uses the integer-scaled IV;
    # MIP uses unscaled IV, so each is checked against its own objective.
    optimum = max(feasible.values())
    if solver == 'cp':
        assert feasible[selected] == optimum
        # The solver reports the integer objective through a floating API.
        assert model.solver_.ObjectiveValue() == pytest.approx(
            optimum, rel=0, abs=1e-7)
    else:
        assert feasible[selected] == pytest.approx(
            optimum, rel=1e-8, abs=1e-9)
