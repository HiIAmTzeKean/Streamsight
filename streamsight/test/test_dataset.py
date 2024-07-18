import os
import pandas as pd
import pytest
from streamsight.datasets import AmazonMusicDataset
from streamsight.matrix import InteractionMatrix


@pytest.fixture()
def dataset_path():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")

class TestAmazonMusicDataset:
    def __init__(self, dataset_path) -> None:
        filename = "amazon_digitalmusic_dataset.csv"
        self.d = AmazonMusicDataset(base_path=dataset_path, filename=filename)
    
    def test_amazon_music_load_dataframe(self):
        
        df = self.d._load_dataframe()
        assert type(df) == pd.DataFrame
        assert (df.columns == [self.d.ITEM_IX, self.d.USER_IX, self.d.RATING_IX, self.d.TIMESTAMP_IX]).all()
        
    def test_amazon_music_load(self):
        data = self.d.load()
        assert type(data) == InteractionMatrix
    