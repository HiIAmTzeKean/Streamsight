import logging
from typing import Dict, List, Literal, Optional, Union

from streamsight.evaluator.evaluator import Evaluator
from streamsight.matrix import ItemUserBasedEnum
from streamsight.registries.registry import (ALGORITHM_REGISTRY,
                                             METRIC_REGISTRY, AlgorithmEntry,
                                             MetricEntry)
from streamsight.setting.base_setting import Setting
from streamsight.utils.util import arg_to_str

logger = logging.getLogger(__name__)

class EvaluatorBuilder(object):
    """Builder to facilitate construction of evaluator.
    Provides methods to set specific values for the evaluator and enforce checks
    such that the evaluator can be constructed correctly and to avoid possible errors
    when the evaluator is executed.
    """

    def __init__(self, item_user_based: Union[Literal["item","user"],ItemUserBasedEnum]):
        if not ItemUserBasedEnum.has_value(item_user_based):
            raise ValueError(f"{item_user_based} invalid value for item_user_based. Value should be in {ItemUserBasedEnum._member_names_}")
        self.item_user_based = ItemUserBasedEnum(item_user_based)
        
        self.algorithm_entries: List[AlgorithmEntry] = []
        """List of algorithms to evaluate"""
        self.metric_entries: Dict[str, MetricEntry] = dict()
        """Dict of metrics to evaluate algorithm on.
        Using Dict instead of List for fast lookup"""
        self.setting: Setting

    def add_algorithm(self, algorithm: Union[str, type], params: Optional[Dict[str, int]] = None):
        algorithm = arg_to_str(algorithm)

        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {algorithm} could not be resolved.")

        if ALGORITHM_REGISTRY.get(algorithm).ITEM_USER_BASED != self.item_user_based:
            raise ValueError(f"Algorithm {algorithm} is not compatible with {self.item_user_based} setting.")
        
        self.algorithm_entries.append(AlgorithmEntry(algorithm, params or {}))

    def add_metric(self, metric: Union[str, type], K: Optional[int] = None):
        metric = arg_to_str(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        metric_name = f"{metric}_{K}"
        if metric_name in self.metric_entries:
            logger.warning(f"Metric {metric_name} already exists."
                           "Skipping adding metric.")
            return

        self.metric_entries[metric_name] = MetricEntry(metric, K)

    def add_setting(self, setting: Setting):
        self.setting = setting

    def _check_ready(self):
        if len(self.metric_entries) == 0:
            raise RuntimeError(
                "No metrics specified, can't construct Evaluator")

        if len(self.algorithm_entries) == 0:
            raise RuntimeError(
                "No algorithms specified, can't construct Evaluator")

        # Check for settings #
        if self.setting is None:
            raise RuntimeError(
                "No settings specified, can't construct Evaluator")
        if not self.setting.is_ready:
            raise RuntimeError(
                "Setting is not ready, can't construct Evaluator. "
                "Call split() on the setting first.")

    def build(self):
        self._check_ready()
        return Evaluator(self.algorithm_entries,
                         list(self.metric_entries.values()),
                         self.setting)
