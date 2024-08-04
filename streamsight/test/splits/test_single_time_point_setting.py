import pandas as pd
import pytest

from streamsight.matrix import InteractionMatrix
from streamsight.matrix.interaction_matrix import ItemUserBasedEnum
from streamsight.setting.base_setting import Setting
from streamsight.setting.single_time_point_setting import SingleTimePointSetting

BACKGROUND_T = 4
SEED = 42
N_SEQ_DATA=1
ITEM_USER_BASED=ItemUserBasedEnum.USER

@pytest.fixture()
def splitter_user():
    return SingleTimePointSetting(background_t=BACKGROUND_T,
                                  n_seq_data=N_SEQ_DATA,
                                  item_user_based=ITEM_USER_BASED,
                                  seed=SEED)

class TestSingleTimePointSettingForUser():
    def test_seed_value(self, splitter_user):
        assert splitter_user.seed == SEED
    
    def test_background_t_value(self, splitter_user):
        assert splitter_user.t == BACKGROUND_T
    
    def test_background_data(self, splitter_user:Setting, matrix:InteractionMatrix):
        expected_background_data = matrix.timestamps_lt(BACKGROUND_T)
        
        splitter_user.split(matrix)
        assert splitter_user.background_data is not None
        assert splitter_user.background_data._df.equals(expected_background_data._df)
    
    def test_ground_truth_data(self, splitter_user:Setting, matrix:InteractionMatrix):
        expected_ground_truth = pd.DataFrame({
            "ts":   [4, 6, 7, 9],
            "uid":  [2, 4, 3, 5],
            "iid":  [2, 2, 1, 1]
        })
        splitter_user.split(matrix)
        assert splitter_user.ground_truth_data is not None
        actual_ground_truth = splitter_user.ground_truth_data._df[["ts","uid","iid"]].reset_index(drop=True)
        assert actual_ground_truth.equals(expected_ground_truth)

    def test_unlabeled_data(self, splitter_user:Setting, matrix:InteractionMatrix):
        expected_unlabeled_data = pd.DataFrame({
            "ts":   [1, 2, 4, 6, 7, 9],
            "uid":   [2, 3, 2, 4, 3, 5],
            "iid":   [1, 2, -1, -1, -1, -1],
        })
        splitter_user.split(matrix)
        assert splitter_user.unlabeled_data is not None
        actual_unlabeled_data = splitter_user.unlabeled_data._df[["ts","uid","iid"]].reset_index(drop=True)
        assert actual_unlabeled_data.equals(expected_unlabeled_data)