import pandas as pd
import pytest

from streamsight.matrix import InteractionMatrix


@pytest.fixture()
def dataframe():
    return pd.DataFrame({
        "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
        "user": [1, 1, 2, 3, 2, 2, 4, 3, 3, 4, 5],
        "item": [1, 3, 1, 2, 2, 3, 2, 1, 3, 3, 1],
        "rating": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })

@pytest.fixture()
def matrix(dataframe):
    return InteractionMatrix(dataframe, "item", "user", "time")
