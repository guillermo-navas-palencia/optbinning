import numpy as np
import pytest
from optbinning import OptimalBinning

@pytest.mark.parametrize('scale',[.13,.3,1,10])
@pytest.mark.parametrize('solver',['cp','mip'])
def test_weighted_mass_and_rates(scale,solver,tmp_path):
    x=np.repeat([0.,1.],20);y=np.r_[np.zeros(15),np.ones(5),np.zeros(5),np.ones(15)]
    m=OptimalBinning(solver=solver,user_splits=[.5],user_splits_fixed=[True],monotonic_trend=None).fit(x,y,sample_weight=np.full(40,scale))
    np.testing.assert_allclose(m.transform([0,1],metric='event_rate'),[.25,.75])
    np.testing.assert_allclose(m.binning_table.n_event.sum(),20*scale)
    np.testing.assert_allclose(m.binning_table.n_nonevent.sum(),20*scale)
    path=tmp_path/'weights.json';m.to_json(str(path));r=OptimalBinning();r.read_json(str(path))
    np.testing.assert_allclose(r.transform([0,1]),m.transform([0,1]))


def test_fractional_cp_class_bounds_forbid_infeasible_split():
    x=np.repeat([0.,1.],20);y=np.r_[np.zeros(15),np.ones(5),np.zeros(5),np.ones(15)]
    m=OptimalBinning(min_bin_n_event=2,user_splits=[.5],user_splits_fixed=[True],monotonic_trend=None).fit(x,y,sample_weight=np.full(40,.3))
    assert m.status=='INFEASIBLE'


def test_fractional_gamma_rejected_explicitly():
    with pytest.raises(ValueError,match='Fractional weights'):
        OptimalBinning(gamma=.1,user_splits=[.5]).fit(np.repeat([0,1],20),np.tile([0,1],20),sample_weight=np.full(40,.13))


def test_fractional_relative_support_and_special_missing():
    x=np.r_[np.repeat([0.,1.],20),[-99,-99,np.nan,np.nan]]
    y=np.r_[np.zeros(15),np.ones(5),np.zeros(5),np.ones(15),0,1,0,1]
    m=OptimalBinning(min_bin_size=.4,max_bin_size=.6,special_codes=[-99],
                     user_splits=[.5],user_splits_fixed=[True],monotonic_trend=None).fit(x,y,sample_weight=np.full(44,.13))
    assert m.status=='OPTIMAL'
    np.testing.assert_allclose(m.binning_table.n_event.sum(),22*.13)
    np.testing.assert_allclose(m.transform([0,1,-99,np.nan],metric='event_rate',metric_special='empirical',metric_missing='empirical'),[.25,.75,.5,.5])


@pytest.mark.parametrize('solver',['cp','mip'])
@pytest.mark.parametrize('offset',[0.,1e-8])
def test_equal_fractional_bound_and_real_violation(solver,offset):
    x=np.repeat([0.,1.,2.],30);y=np.tile([0,1],45)
    m=OptimalBinning(solver=solver,user_splits=[.5,1.5],user_splits_fixed=[True,True],
        monotonic_trend=None,min_bin_size=1/3+offset,max_bin_size=None if offset else 1/3).fit(x,y,sample_weight=np.full(90,.13))
    assert m.status==('OPTIMAL' if offset==0 else 'INFEASIBLE')


@pytest.mark.parametrize('class_weight',[{0:.2,1:1.3},'balanced'])
def test_cart_uses_effective_weights_once(monkeypatch,class_weight):
    import optbinning.binning.binning as module
    from sklearn.utils.class_weight import compute_sample_weight
    rng=np.random.default_rng(832)
    x=rng.normal(size=500);y=rng.binomial(1,1/(1+np.exp(-x)))
    w=rng.uniform(.1,2,500);effective=w*compute_sample_weight(class_weight,y)
    original=module.PreBinning
    def checked(*args,**kwargs):
        assert kwargs.get('class_weight') is None
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'PreBinning',checked)
    actual=OptimalBinning(class_weight=class_weight).fit(x,y,sample_weight=w)
    expected=OptimalBinning().fit(x,y,sample_weight=effective)
    np.testing.assert_array_equal(actual.splits,expected.splits)
    np.testing.assert_allclose(actual.binning_table.n_event,expected.binning_table.n_event)


def test_cbc_fractional_support_without_pure_bins():
    x=np.repeat([0.,1.],20);y=np.r_[np.zeros(15),np.ones(5),np.zeros(5),np.ones(15)]
    model=OptimalBinning(solver='mip',mip_solver='cbc',min_bin_size=.5,max_bin_size=.5,
        user_splits=[.5],user_splits_fixed=[True],monotonic_trend=None).fit(x,y,sample_weight=np.full(40,.13))
    assert model.status=='OPTIMAL'


@pytest.mark.parametrize("weights", [
    [-1, 2, 2, 2], [0, 0, 0, 0], [np.nan, 1, 1, 1],
    [np.inf, 1, 1, 1], [1e308] * 4,
])
def test_invalid_effective_weights_rejected(weights):
    with pytest.raises(ValueError, match="Effective sample weights"):
        OptimalBinning(user_splits=[]).fit(
            [0, 1, 2, 3], [0, 1, 0, 1], sample_weight=weights)


@pytest.mark.parametrize("weights", [[[1], [1], [1], [1]], [1, 2]])
def test_weights_cannot_broadcast_against_class_weights(weights):
    with pytest.raises(ValueError, match="sample_weight must match"):
        OptimalBinning(class_weight={0: .5, 1: 1.5}).fit(
            [0, 1, 2, 3], [0, 1, 0, 1], sample_weight=weights)


def test_negative_class_mass_rejected():
    with pytest.raises(ValueError, match="Effective sample weights"):
        OptimalBinning(class_weight={0: -1, 1: 2}, user_splits=[]).fit(
            [0, 1, 2, 3], [0, 1, 0, 1])


def test_scalar_and_vector_weight_equivalence_without_mutation():
    x = np.arange(4); y = np.array([0, 1, 0, 1])
    weights = np.full(4, .3)
    a = OptimalBinning(user_splits=[]).fit(x, y, sample_weight=.3)
    b = OptimalBinning(user_splits=[]).fit(x, y, sample_weight=weights)
    np.testing.assert_allclose(a.binning_table.n_event, b.binning_table.n_event)
    np.testing.assert_array_equal(weights, np.full(4, .3))
