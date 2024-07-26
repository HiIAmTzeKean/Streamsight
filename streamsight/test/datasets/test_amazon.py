import os
import pandas as pd
import pytest
from streamsight.datasets import AmazonMusicDataset
from streamsight.datasets.base import Dataset
from streamsight.matrix import InteractionMatrix

@pytest.fixture()
def dataset_path(filename):
    return os.path.join(os.getcwd(),"streamsight","test","data")

@pytest.fixture()
def filename():
    return "amazon_digitalmusic_dataset.csv"

@pytest.fixture()
def dataset(dataset_path, filename):
    return AmazonMusicDataset(
            base_path=dataset_path, filename=filename)

class TestAmazonMusicDataset:
    d: Dataset

    def test_amazon_music_load_dataframe(self, dataset):
        df = dataset._load_dataframe()
        assert type(df) == pd.DataFrame
        assert (df.columns == [dataset.ITEM_IX, dataset.USER_IX,
                dataset.RATING_IX, dataset.TIMESTAMP_IX]).all()

    def test_amazon_music_load(self, dataset):
        data = dataset.load()
        assert type(data) == InteractionMatrix
        