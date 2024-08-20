import pandas as pd
import pytest

from streamsight.datasets import Dataset, TestDataset


@pytest.fixture()
def test_dataframe() -> pd.DataFrame:
    return pd.DataFrame({
        "time": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 10, 10],
        "user":   [1, 2, 3, 1, 2, 2, 4, 3, 3, 4, 5, 5, 5],
        "item":   [1, 1, 2, 3, 2, 3, 2, 1, 3, 3, 1, 2, 3],
        "rating": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })

@pytest.fixture()
def test_dataset() -> Dataset:
    dataset = TestDataset()
    return dataset