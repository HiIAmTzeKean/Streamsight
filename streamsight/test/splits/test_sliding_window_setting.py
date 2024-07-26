import pytest

from streamsight.matrix.interation_matrix import InteractionMatrix
from streamsight.splits.base_setting import Setting
from streamsight.splits.sliding_window_setting import SlidingWindowSetting
from streamsight.splits.util import FrameExpectedError


BACKGROUND_T = 4
WINDOW_SIZE = 3
SEED = 42

@pytest.fixture()
def splitter():
    return SlidingWindowSetting(background_t=BACKGROUND_T, window_size=WINDOW_SIZE, seed=SEED)


class TestSlidingWindowSetting():
    def test_seed_value(self, splitter):
        assert splitter.seed == SEED

    def test_background_t_value(self, splitter):
        assert splitter.t == BACKGROUND_T
        
    def test_split_properties(self, splitter: Setting,  matrix: InteractionMatrix):
        splitter._split(matrix)
        assert splitter._num_split_set == 2

    def test_background_data(self, splitter: Setting, matrix: InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(BACKGROUND_T)

        splitter.split(matrix)
        assert splitter.background_data is not None
        assert splitter.background_data._df.equals(
            expected_background_data._df)

    def test_ground_truth_data(self, splitter: Setting, matrix: InteractionMatrix):
        splitter._split(matrix)
        print(splitter.ground_truth_data_frame[0])
        print("test")

    def test_data_time_limit(self, splitter: Setting, matrix: InteractionMatrix):
        splitter._split(matrix)
        assert splitter.data_timestamp_limit == [4, 7]
    def test_frame_error_thrown(self, splitter: Setting, matrix: InteractionMatrix):
        with pytest.raises(FrameExpectedError) as e_info:
            splitter.ground_truth_data_series
        with pytest.raises(FrameExpectedError) as e_info:
            splitter.unlabeled_data_series
