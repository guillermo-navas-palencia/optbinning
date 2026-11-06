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


# Keep this separate from automatic target detection: sklearn's generated
# integer-valued targets also have a valid continuous-target interpretation.
# This exercises the continuous binning implementation and its weight handling
# without modifying sklearn's test data or overriding estimator tags.
@parametrize_with_checks([BinningProcess(target_dtype="continuous")])
def test_continuous_binning_process_estimator_contract(estimator, check):
    check(estimator)
