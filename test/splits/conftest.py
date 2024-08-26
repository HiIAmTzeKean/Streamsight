import pandas as pd
import pytest

from streamsight.matrix import InteractionMatrix
from test.conftest import test_dataframe

@pytest.fixture()
def matrix(test_dataframe):
    return InteractionMatrix(test_dataframe, "item", "user", "time")
