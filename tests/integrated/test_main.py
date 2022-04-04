import pytest
from ts_boilerplate.main import train, cross_validate

@pytest.mark.slow
def test_main_route_train(data_zeros_and_ones):
    train(data_zeros_and_ones)

@pytest.mark.slow
def test_main_route_cross_validate(data_zeros_and_ones):
    cross_validate(data_zeros_and_ones)
