import logging
from typing import List

from streamsight.evaluator.accumulator import (MacroMetricAccumulator,
                                               MicroMetricAccumulator)
from streamsight.evaluator.util import UserItemBaseStatus
from streamsight.registries import (MetricEntry)
from streamsight.settings.base import Setting

logger = logging.getLogger(__name__)

class EvaluatorBase(object):
    """Base class for evaluator.
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
        self.ignore_unknown_user = ignore_unknown_user
        self.ignore_unknown_item = ignore_unknown_item
        
        self._micro_acc: MicroMetricAccumulator
        self._macro_acc: MacroMetricAccumulator
        self.user_item_base = UserItemBaseStatus()
        
        self.ignore_unknown_user = ignore_unknown_user
        self.ignore_unknown_item = ignore_unknown_item
        
        self._run_step = 0