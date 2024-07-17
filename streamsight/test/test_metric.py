
from streamsight.metrics.precision import PrecisionK


test_metric = PrecisionK(10,123)

def test_class_name():
    assert test_metric.name == "PrecisionK_10"

def test_class_identifier():
    assert test_metric.identifier == "PrecisionK(t<=123,K=10)"