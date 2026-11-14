from training_utils import split_2022_periods


def test_2022_calibration_and_policy_periods_are_chronological_and_disjoint():
    timestamps = [1654041600, 1656633599, 1656633600, 1672444800]
    calibration, policy = split_2022_periods(timestamps)
    assert calibration == [True, True, False, False]
    assert policy == [False, False, True, True]
    assert all(not (left and right)
               for left, right in zip(calibration, policy))
    assert all(left or right for left, right in zip(calibration, policy))


def test_2022_split_rejects_no_values_by_returning_empty_masks():
    calibration, policy = split_2022_periods([])
    assert calibration == []
    assert policy == []
