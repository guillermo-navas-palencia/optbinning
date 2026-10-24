"""Effective class/sample weight contract and wide dynamic-range checks."""
import numpy as np
import pytest
from optbinning import OptimalBinning


def data():
    x=np.repeat([0.,1.,2.],[20,20,40])
    y=np.r_[np.zeros(15),np.ones(5),np.tile([0,1],10),np.zeros(10),np.ones(30)]
    return x,y


def model(solver, **kwargs):
    return OptimalBinning(solver=solver,
        user_splits=[.5,1.5],user_splits_fixed=[True,True],monotonic_trend=None,**kwargs)


@pytest.mark.parametrize('solver',['cp','mip'])
@pytest.mark.parametrize('class_weight',[{0:.2,1:1.3},'balanced'])
def test_effective_mass_and_no_caller_mutation(solver,class_weight):
    x,y=data();w=np.tile([.7,1.3],40);before=w.copy()
    factors={0:.2,1:1.3} if isinstance(class_weight,dict) else {0:80/70,1:80/90}
    effective=w*np.array([factors[v] for v in y])
    m=model(solver,class_weight=class_weight,min_bin_size=.05,max_bin_size=.8).fit(x,y,sample_weight=w)
    assert m.status=='OPTIMAL'
    np.testing.assert_array_equal(w,before)
    expected_e=[effective[(x==v)&(y==1)].sum() for v in [0,1,2]]
    expected_ne=[effective[(x==v)&(y==0)].sum() for v in [0,1,2]]
    np.testing.assert_allclose(m.binning_table.n_event[:3],expected_e)
    np.testing.assert_allclose(m.binning_table.n_nonevent[:3],expected_ne)
    np.testing.assert_allclose(m._n_samples_weighted,effective.sum())
    np.testing.assert_allclose(m.transform([0,1,2],metric='event_rate'),np.array(expected_e)/(np.array(expected_e)+expected_ne))


@pytest.mark.parametrize('solver',['cp','mip'])
def test_class_weights_fractional_gamma_guard(solver):
    x,y=data()
    with pytest.raises(ValueError,match='Fractional weights'):
        model(solver,class_weight={0:.2,1:1.3},gamma=.1).fit(x,y)


@pytest.mark.parametrize('solver',['cp','mip'])
@pytest.mark.parametrize('scale',[1e-12,1e-8,1e8,1e12])
def test_extreme_positive_scale_preserves_mass_rate_and_support(solver,scale):
    x,y=data();w=np.full(80,scale)
    m=model(solver,min_bin_size=.2,max_bin_size=.6).fit(x,y,sample_weight=w)
    assert m.status=='OPTIMAL'
    np.testing.assert_allclose(m.binning_table.n_event[:3],np.array([5,10,30])*scale,rtol=1e-12,atol=0)
    np.testing.assert_allclose(m.binning_table.n_nonevent[:3],np.array([15,10,10])*scale,rtol=1e-12,atol=0)
    np.testing.assert_allclose(m.transform([0,1,2],metric='event_rate'),[.25,.5,.75],rtol=1e-12,atol=0)

@pytest.mark.parametrize('solver',['cp','mip'])
def test_effective_support_rejects_mass_exceeding_maximum(solver):
    x,y=data();w=np.tile([.7,1.3],40)
    effective=w*np.where(y==0,.2,1.3)
    assert effective[x==2].sum()/effective.sum() > .55
    assert w[x==2].sum()/w.sum() <= .55
    m=model(solver,class_weight={0:.2,1:1.3},max_bin_size=.55).fit(x,y,sample_weight=w)
    assert m.status=='INFEASIBLE'
