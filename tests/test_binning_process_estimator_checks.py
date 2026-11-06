"""Run sklearn's general estimator contract checks on BinningProcess.

Each generated check is a separate pytest case, so failures can be reproduced
and addressed individually. Keep the default estimator and its declared tags
unchanged: this suite audits the public interface, not an adapted wrapper.
"""

from sklearn.utils.estimator_checks import parametrize_with_checks

from optbinning import BinningProcess


@parametrize_with_checks([BinningProcess()])
def test_binning_process_estimator_contract(estimator, check):
    check(estimator)
