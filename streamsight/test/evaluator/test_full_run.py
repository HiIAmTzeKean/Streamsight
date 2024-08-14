import pytest
from streamsight.datasets import TestDataset
from streamsight.settings import SlidingWindowSetting, SingleTimePointSetting
from streamsight.evaluator.evaluator_builder import EvaluatorBuilder

@pytest.fixture()
def sliding_window():
    dataset = TestDataset()
    data = dataset.load()
    setting = SlidingWindowSetting(
        4,
        3
    )
    setting.split(data)
    return setting

@pytest.fixture()
def single_time_point():
    dataset = TestDataset()
    data = dataset.load()
    setting = SingleTimePointSetting(
        4,
    )
    setting.split(data)
    return setting

class TestFullRun:
    def test_sliding_window(self, sliding_window):
        b = EvaluatorBuilder()
        b.add_algorithm("ItemKNNIncremental", {"K": 1})
        b.add_metric("PrecisionK")
        b.add_metric("RecallK")
        b.add_setting(sliding_window)
        evaluator = b.build()
        evaluator.run()
        
    def test_single_time_point(self, single_time_point):
        b = EvaluatorBuilder()
        b.add_algorithm("ItemKNNIncremental", {"K": 1})
        b.add_metric("PrecisionK")
        b.add_metric("RecallK")
        b.add_setting(single_time_point)
        evaluator = b.build()
        evaluator.run()