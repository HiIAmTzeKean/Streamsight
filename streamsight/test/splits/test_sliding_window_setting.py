import pytest

from streamsight.matrix import InteractionMatrix
from streamsight.matrix.interaction_matrix import ItemUserBasedEnum
from streamsight.setting.base_setting import Setting
from streamsight.setting.sliding_window_setting import SlidingWindowSetting


BACKGROUND_T = 4
WINDOW_SIZE = 3
SEED = 42
N_SEQ_DATA=1
ITEM_USER_BASED=ItemUserBasedEnum.USER

@pytest.fixture()
def splitter():
    return SlidingWindowSetting(background_t=BACKGROUND_T,
                                window_size=WINDOW_SIZE,
                                n_seq_data=N_SEQ_DATA,
                                item_user_based=ITEM_USER_BASED,
                                seed=SEED)


class TestSlidingWindowSetting():
    def test_seed_value(self, splitter):
        assert splitter.seed == SEED

# TODO test if the overall global user and item id are correctly masked
    def test_background_t_value(self, splitter):
        assert splitter.t == BACKGROUND_T
        
    def test_split_properties(self, splitter: Setting,  matrix: InteractionMatrix):
        splitter._split(matrix)
        assert splitter._num_split_set == 2
        
    def test_access_split_attributes_before_split(self, splitter: Setting):
        with pytest.raises(KeyError) as e_info:
            splitter.background_data
        with pytest.raises(KeyError) as e_info:
            splitter.unlabeled_data
        with pytest.raises(KeyError) as e_info:
            splitter.ground_truth_data
        with pytest.raises(KeyError) as e_info:
            splitter.incremental_data
            
    def test_background_data(self, splitter: Setting, matrix: InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(BACKGROUND_T)

        splitter.split(matrix)
        assert splitter.background_data is not None
        assert splitter.background_data._df.equals(
            expected_background_data._df)

    def test_incremental_data(self, splitter: Setting, matrix: InteractionMatrix):
        expected_incremental_data_0 = matrix.timestamps_gte(BACKGROUND_T).timestamps_lt(BACKGROUND_T + WINDOW_SIZE)
        expected_ground_truth_data_1 = matrix.timestamps_gte(BACKGROUND_T + WINDOW_SIZE).timestamps_lt(BACKGROUND_T + 2*WINDOW_SIZE)
        splitter.split(matrix)
        assert splitter.incremental_data[0]._df.equals(
            expected_incremental_data_0._df)
        assert splitter.incremental_data[1]._df.equals(
            expected_ground_truth_data_1._df)
        
    def test_data_time_limit(self, splitter: Setting, matrix: InteractionMatrix):
        splitter.split(matrix)
        assert splitter.data_timestamp_limit == [4, 7]
