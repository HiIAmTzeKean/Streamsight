import logging
from typing import Dict, List, Literal, Optional, Union
from warnings import warn

from streamsight.evaluator.evaluator import Evaluator
from streamsight.matrix import ItemUserBasedEnum
from streamsight.registries.registry import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
)
from streamsight.settings.base_setting import Setting
from streamsight.utils.util import arg_to_str

logger = logging.getLogger(__name__)


class EvaluatorBuilder(object):
    """Builder to facilitate construction of evaluator.
    Provides methods to set specific values for the evaluator and enforce checks
    such that the evaluator can be constructed correctly and to avoid possible
    errors when the evaluator is executed.
    
    :param ignore_unknown_user: Ignore unknown user in the evaluation, defaults to True
    :type ignore_unknown_user: bool, optional
    :param ignore_unknown_item: Ignore unknown item in the evaluation, defaults to True
    :type ignore_unknown_item: bool, optional
    """
    def __init__(
        self,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
    ):
        self.algorithm_entries: List[AlgorithmEntry] = []
        """List of algorithms to evaluate"""
        self.metric_entries: Dict[str, MetricEntry] = dict()
        """Dict of metrics to evaluate algorithm on.
        Using Dict instead of List for fast lookup"""
        self.setting: Setting
        """Setting to evaluate the algorithms on"""
        self.ignore_unknown_user = ignore_unknown_user
        """Ignore unknown user in the evaluation"""
        self.ignore_unknown_item = ignore_unknown_item
        """Ignore unknown item in the evaluation"""

    def add_algorithm(
        self,
        algorithm: Union[str, type],
        params: Optional[Dict[str, int]] = None,
    ):
        """Add algorithm to evaluate.

        Adding algorithm to evaluate on. The algorithm can be added by specifying the class type
        or by specifying the class name as a string.

        :param algorithm: Algorithm to evaluate
        :type algorithm: Union[str, type]
        :param params: Parameter for the algorithm, defaults to None
        :type params: Optional[Dict[str, int]], optional
        :raises ValueError: If algorithm is not found in ALGORITHM_REGISTRY
        """
        algorithm = arg_to_str(algorithm)

        if algorithm not in ALGORITHM_REGISTRY:
            raise ValueError(f"Algorithm {algorithm} could not be resolved.")

        self.algorithm_entries.append(AlgorithmEntry(algorithm, params or {}))

    def add_metric(
        self, metric: Union[str, type], K: Optional[int] = None
    ) -> None:
        """Add metric to evaluate algorithm on.

        :param metric: Metric to evaluate algorithm on
        :type metric: Union[str, type]
        :param K: Top K value to evaluate the prediction on, defaults to None
        :type K: Optional[int], optional
        :raises ValueError: If metric is not found in METRIC_REGISTRY
        """
        metric = arg_to_str(metric)

        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric {metric} could not be resolved.")

        metric_name = f"{metric}_{K}"
        if metric_name in self.metric_entries:
            logger.warning(
                f"Metric {metric_name} already exists."
                "Skipping adding metric."
            )
            return

        self.metric_entries[metric_name] = MetricEntry(metric, K)

    def add_setting(self, setting: Setting) -> None:
        """Add setting to the evaluator builder.

        :param setting: Setting to evaluate the algorithms on
        :type setting: Setting
        :raises ValueError: If setting is not of instance Setting
        """
        if not isinstance(setting, Setting):
            raise ValueError(
                f"setting should be of type Setting, got {type(setting)}"
            )
        self.setting = setting

    def _check_ready(self):
        """Check if the builder is ready to construct Evaluator.

        :raises RuntimeError: If there are invalid configurations
        """
        if len(self.metric_entries) == 0:
            raise RuntimeError(
                "No metrics specified, can't construct Evaluator"
            )

        if len(self.algorithm_entries) == 0:
            raise RuntimeError(
                "No algorithms specified, can't construct Evaluator"
            )

        # Check for settings #
        if self.setting is None:
            raise RuntimeError(
                "No settings specified, can't construct Evaluator"
            )
        if not self.setting.is_ready:
            raise RuntimeError(
                "Setting is not ready, can't construct Evaluator. "
                "Call split() on the setting first."
            )

        for algo in self.algorithm_entries:
            if (
                algo.params is not None
                and "K" in algo.params
                and algo.params["K"] != self.setting.top_K
            ):
                warn(
                    f"Algorithm {algo.name} has K={algo.params['K']} but setting"
                    f" is configured top_K={self.setting.top_K}. Mismatch in K will"
                    " cause metric to evaluate on algorithm's K value but number of"
                    " prediction requested from model will mismatch that K value"
                )

    def build(self) -> Evaluator:
        """Build Evaluator object.

        :raises RuntimeError: If no metrics, algorithms or settings are specified
        :return: Evaluator object
        :rtype: Evaluator
        """
        self._check_ready()
        return Evaluator(
            algorithm_entries=self.algorithm_entries,
            metric_entries=list(self.metric_entries.values()),
            setting=self.setting,
            ignore_unknown_user=self.ignore_unknown_user,
            ignore_unknown_item=self.ignore_unknown_item,
        )
