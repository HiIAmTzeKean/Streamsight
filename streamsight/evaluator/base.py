import logging
from typing import List, Literal, Optional, Union

import pandas as pd

from streamsight.evaluator.accumulator import (MacroMetricAccumulator,
                                               MicroMetricAccumulator)
from streamsight.evaluator.util import MetricLevelEnum, UserItemBaseStatus
from streamsight.registries import (MetricEntry)
from streamsight.settings.base import Setting

logger = logging.getLogger(__name__)

class EvaluatorBase(object):
    """Base class for evaluator.
    
    :param metric_entries: List of metric entries to compute
    :type metric_entries: List[MetricEntry]
    :param setting: Setting object
    :type setting: Setting
    :param ignore_unknown_user: Ignore unknown users, defaults to True
    :type ignore_unknown_user: bool, optional
    :param ignore_unknown_item: Ignore unknown items, defaults to True
    :type ignore_unknown_item: bool, optional
    """
    def __init__(
        self,
        metric_entries: List[MetricEntry],
        setting: Setting,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
    ):
        self.metric_entries = metric_entries
        self.setting = setting
        """Setting to evaluate the algorithms on."""
        self.ignore_unknown_user = ignore_unknown_user
        """To ignore unknown users during evaluation."""
        self.ignore_unknown_item = ignore_unknown_item
        """To ignore unknown items during evaluation."""
        
        self._micro_acc: MicroMetricAccumulator
        self._macro_acc: MacroMetricAccumulator
        self.user_item_base = UserItemBaseStatus()
        
        self.ignore_unknown_user = ignore_unknown_user
        self.ignore_unknown_item = ignore_unknown_item
        
        self._run_step = 0
        self._current_timestamp: int
        
    def metric_results(self,
                       level:Union[Literal["micro","macro"], MetricLevelEnum]="macro",
                       only_current_timestamp=False,
                       filter_algo:Optional[str]=None) -> pd.DataFrame:
        """Results of the metrics computed.
        
        Computes the metrics of all algorithms based on the level specified and
        return the results in a pandas DataFrame. The results can be filtered
        based on the algorithm name and the current timestamp.
        
        If the level specified is "macro", then the filtering on the current timestamp
        will be ignored.
        """
        timestamp = None
        if only_current_timestamp:
            timestamp = self._current_timestamp

        acc = self._macro_acc
        if level == MetricLevelEnum.MICRO:
            acc = self._micro_acc
        
        return acc.df_metric(filter_algo=filter_algo,
                             filter_timestamp=timestamp)