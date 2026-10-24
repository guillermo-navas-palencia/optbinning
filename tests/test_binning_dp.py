"""Exact objective oracle and public-estimator guarded dispatch checks."""
import itertools
import pickle
import numpy as np
import pytest
from sklearn.base import clone
from optbinning import OptimalBinning
from optbinning.binning.dp import solve_iv


@pytest.mark.parametrize('seed',range(8))
def test_dp_matches_exhaustive_integer_objective(seed):
    rng=np.random.default_rng(seed)
    ne=rng.integers(5,40,6);ev=rng.integers(5,40,6)
    result=solve_iv(ne,ev,2,4,25,160,10)
    values=[]
    for k in range(2,5):
        for cuts in itertools.combinations(range(1,6),k-1):
            bounds=(0,)+cuts+(6,);score=0
            for start,stop in zip(bounds,bounds[1:]):
                e,n=ev[start:stop].sum(),ne[start:stop].sum()
                if not 25<=e+n<=160:break
                p,q=e/ev.sum(),n/ne.sum()
                score+=int(1e6*(p-q)*np.log(p/q))
            else:values.append(score)
    assert result[0]=='OPTIMAL'
    assert result[2]==max(values)
    assert 2<=result[1].sum()<=4
    bounds=np.r_[0,np.flatnonzero(result[1])+1]
    actual=0
    for start,stop in zip(bounds,bounds[1:]):
        e,n=ev[start:stop].sum(),ne[start:stop].sum()
        assert 25<=e+n<=160
        p,q=e/ev.sum(),n/ne.sum()
        actual+=int(1e6*(p-q)*np.log(p/q))
    assert actual==result[2]


def data():
    rng=np.random.default_rng(821)
    x=rng.normal(size=600)
    y=rng.binomial(1,.1+.7*(x>0))
    return x,y


def test_public_auto_matches_cp_and_survives_pickle(capsys):
    x,y=data()
    kw=dict(monotonic_trend=None,min_n_bins=2,max_n_bins=4,min_bin_size=.1)
    cp=OptimalBinning(solver='cp',**kw).fit(x,y)
    fast=OptimalBinning(solver='auto',**kw).fit(x,y)
    assert fast.solver_backend_=='dp'
    assert fast.status==cp.status=='OPTIMAL'
    assert fast._optimizer['objective']==cp._optimizer['objective']
    fast.information()
    assert 'dp' in capsys.readouterr().out
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(fast)).transform(x),fast.transform(x))
    assert clone(fast).solver=='auto'


@pytest.mark.parametrize('extra',[dict(monotonic_trend='ascending'),dict(gamma=.1),
    dict(max_pvalue=.1),dict(min_event_rate_diff=.05),dict(min_bin_n_event=5),
    dict(divergence='js'),dict(user_splits=[0.],user_splits_fixed=[True])])
def test_unsupported_constraints_fall_back_to_cp(extra):
    x,y=data();kw=dict(monotonic_trend=None);kw.update(extra)
    cp=OptimalBinning(solver='cp',**kw).fit(x,y)
    auto=OptimalBinning(solver='auto',**kw).fit(x,y)
    assert auto.solver_backend_=='cp'
    assert auto.status==cp.status
    np.testing.assert_array_equal(auto.splits,cp.splits)


def test_deadline_infeasibility_and_fractional_guard():
    assert solve_iv([2.5,3],[3,4],1,2,None,None,10) is None
    assert solve_iv([2,3],[3,4],1,2,None,None,1e-12) is None
    result=solve_iv([2,3],[3,4],2,2,10,None,10)
    assert result[0]=='INFEASIBLE' and result[2] is None


def test_dispatch_timeout_falls_back(monkeypatch):
    import optbinning.binning.binning as module
    monkeypatch.setattr(module,'solve_iv',lambda *a:None)
    x,y=data()
    model=OptimalBinning(solver='auto',monotonic_trend=None).fit(x,y)
    assert model.solver_backend_=='cp'


def test_categorical_auto_preserves_cp_fallback():
    x=np.tile(['a','b','c'],100);y=np.tile([0,0,1,1,0,1],50)
    model=OptimalBinning(dtype='categorical',solver='auto',monotonic_trend=None).fit(x,y)
    assert model.solver_backend_=='cp'
