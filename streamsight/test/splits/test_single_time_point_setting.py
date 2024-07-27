import pytest

from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.splits.base_setting import Setting
from streamsight.splits.single_time_point_setting import SingleTimePointSetting
from streamsight.splits.util import SeriesExpectedError

BACKGROUND_T = 4
SEED = 42

@pytest.fixture()
def splitter():
    return SingleTimePointSetting(background_t=BACKGROUND_T, seed=SEED)

class TestSingleTimePointSetting():
    def test_seed_value(self, splitter):
        assert splitter.seed == SEED
    
    def test_background_t_value(self, splitter):
        assert splitter.t == BACKGROUND_T
    
    def test_background_data(self, splitter:Setting, matrix:InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(BACKGROUND_T)
        
        splitter.split(matrix)
        assert splitter.background_data is not None
        assert splitter.background_data._df.equals(expected_background_data._df)
    
    def test_ground_truth_data(self, splitter:Setting, matrix:InteractionMatrix):
        expected_ground_truth = matrix.timestamps_gte(BACKGROUND_T)
        
        splitter.split(matrix)
        assert splitter.ground_truth_data is not None
        assert splitter.ground_truth_data._df.equals(expected_ground_truth._df)
