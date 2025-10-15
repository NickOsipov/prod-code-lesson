import pytest

from src.preprocessing import summ_array


@pytest.mark.parametrize("input_array, expected_output", [
    ([1, 2, 3], 6),
    ([1.5, 2.5, 3.0], 7.0),
    ([-1, 1, -2, 2], 0),
    ([0, 0, 0], 0),
    ([1], 1)
])
def test_summ_array(input_array, expected_output):
    assert summ_array(input_array) == expected_output
