import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from logging import warn
from typing import Optional

import pandas as pd
from scipy.sparse import csr_matrix

from streamsight.evaluators.util import MetricLevelEnum
from streamsight.metrics import Metric

logger = logging.getLogger(__name__)

class MetricAccumulator(ABC):
    """Accumulates and aggregate :class:`Metric`
    
    Base class for accumulating metrics. The class is used to store and aggregate
    metrics for each algorithm. The class is abstract and should not be instantiated
    directly. Instead, use the :class:`MacroMetricAccumulator` or :class:`MicroMetricAccumulator`
    to accumulate metrics.
    """
    def __init__(self):
        self.acc: defaultdict[str, dict[str, Metric]] = defaultdict(dict)

    def __getitem__(self, key):
        return self.acc[key]

    def add(self, metric: Metric, algorithm_name: str) -> None:
        """Add a metric to the accumulator
        
        Takes a :class:`Metric` object and adds it under the algorithm name. If
        the specified metric already exists for the algorithm, it will be
        overwritten with the new metric.

        :param metric: Metric to store
        :type metric: Metric
        :param algorithm_name: Name of the algorithm
        :type algorithm_name: str
        """
        if metric.identifier in self.acc[algorithm_name]:
            warn(
                f"Metric {metric.identifier} already exists for algorithm {algorithm_name}. Overwriting...")

        logger.debug(
            f"Metric {metric.identifier} created for algorithm {algorithm_name}")
        
        self.acc[algorithm_name][metric.identifier] = metric

    @abstractmethod
    def df_metric(self,
                  filter_timestamp:Optional[int]=None,
                  filter_algo:Optional[str]=None
                  ) -> pd.DataFrame:
        """Dataframe representation of the metric

        :param filter_timestamp: Timestamp value to filter on, defaults to None
        :type filter_timestamp: Optional[int], optional
        :param filter_algo: Algorithm name to filter on, defaults to None
        :type filter_algo: Optional[str], optional
        :return: Dataframe representation of the metric
        :rtype: pd.DataFrame
        """
        pass

    @property
    @abstractmethod
    def metrics(self) -> defaultdict:
        """Compute metric for output
        
        Computes the metric for each algorithm into a dictionary. The Key value
        of the dictionary depends on the level of aggregation of the metric
        and the algorithm name.

        :return: _description_
        :rtype: defaultdict
        """
        pass

class MacroMetricAccumulator(MetricAccumulator):
    def __init__(self):
        super().__init__()
        self.level = MetricLevelEnum.MACRO
        self.ready = False
    
    def _calculate(self):
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                self.acc[algo_name][metric_identifier].calculate_cached()
        self.ready = True
    
    def add(self, metric: Metric, algorithm_name: str) -> None:
        super().add(metric, algorithm_name)
        self.ready = False
        
    def cache_results(self, algo:str, y_true:csr_matrix, y_pred):
        for item in self.acc[algo]:
            self.acc[algo][item].cache_values(y_true,y_pred)
        self.ready = False
    
    @property
    def metrics(self) -> defaultdict:
        if not self.ready:
            self._calculate()
        
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, metric.name)]["score"] = metric.macro_result
        return results
    
    def df_metric(self,
                  filter_timestamp:Optional[int]=None,
                  filter_algo:Optional[str]=None
                  ) -> pd.DataFrame:
        """Macro metric across all timestamps

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.metrics, orient="index")
        df = df.rename_axis(["Algorithm", "Metric"])
        
        # TODO allow filter for macro on specific timestamp?
        if filter_algo:
            df = df.filter(like=filter_algo, axis=0)

        return df

class MicroMetricAccumulator(MetricAccumulator):
    def __init__(self):
        super().__init__()
        self.level = MetricLevelEnum.MICRO
    
    @property
    def user_level_metrics(self):
        results = defaultdict()
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)] = metric.micro_result
        return results

    @property
    def metrics(self) -> defaultdict:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["score"] = metric.macro_result
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)]["num_user"] = metric.num_users
        return results

    def df_user_level_metric(self) -> pd.DataFrame:
        """User metric across all timestamps
        
        Computation of metrics evaluated on the user level

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.user_level_metrics, orient="index").explode(["user_id","score"])
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        return df
    
    def df_metric(self,
                  filter_timestamp:Optional[int]=None,
                  filter_algo:Optional[str]=None
                  ) -> pd.DataFrame:
        df = pd.DataFrame.from_dict(self.metrics, orient="index")
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        
        if filter_algo:
            df = df.filter(like=filter_algo, axis=0)
        if filter_timestamp:
            df = df.filter(like=f"t={filter_timestamp}", axis=0)
        return df