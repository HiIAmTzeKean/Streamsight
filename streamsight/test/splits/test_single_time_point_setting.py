from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.splits.base_setting import Setting
from streamsight.splits.single_time_point_setting import SingleTimePointSetting
import pytest

from streamsight.splits.util import SeriesExpectedError

@pytest.fixture()
def splitter():
    return SingleTimePointSetting(background_t=3, seed=42)

class TestSingleTimePointSetting():
    def test_seed_value(self, splitter):
        assert splitter.seed == 42
    
    def test_background_t_value(self, splitter):
        assert splitter.t == 3
    
    def test_background_data(self, splitter:Setting, matrix:InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(3)
        
        splitter.split(matrix)
        assert splitter.background_data is not None
        assert splitter.background_data._df.equals(expected_background_data._df)
    
    def test_ground_truth_data(self, splitter:Setting, matrix:InteractionMatrix):
        expected_ground_truth = matrix.timestamps_gte(3)
        
        splitter.split(matrix)
        assert splitter.ground_truth_data_series is not None
        assert splitter.ground_truth_data_series._df.equals(expected_ground_truth._df)
        assert splitter.ground_truth_data._df.equals(expected_ground_truth._df)
        
    def test_frame_error_thrown(self, splitter:Setting, matrix:InteractionMatrix):
        with pytest.raises(SeriesExpectedError) as e_info:
            splitter.ground_truth_data_frame
        with pytest.raises(SeriesExpectedError) as e_info:
            splitter.unlabeled_data_frame
            
        