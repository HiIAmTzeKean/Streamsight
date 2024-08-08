import logging
from collections import defaultdict
from enum import StrEnum
from logging import warn
from typing import Literal, Optional, Tuple, Union

import pandas as pd

from streamsight.metrics.base import Metric

logger = logging.getLogger(__name__)

class MetricLevelEnum(StrEnum):
    MICRO = "micro"
    MACRO = "macro"
    
    @classmethod
    def has_value(cls, value: str):
        """Check valid value for MetricLevelEnum

        :param value: String value input
        :type value: str
        """
        if value not in MetricLevelEnum:
            return False
        return True
    
    
class MetricAccumulator:
    """Accumulates metrics and provides methods to aggregate results into usable formats.
    """

    def __init__(self):
        self.acc: defaultdict[str, dict[str, Metric]] = defaultdict(dict)

    def __getitem__(self, key):
        return self.acc[key]

    def add(self, metric: Metric, algorithm_name: str):
        if metric.identifier in self.acc[algorithm_name]:
            warn(
                f"Metric {metric.identifier} already exists for algorithm {algorithm_name}. Overwriting...")

        logger.debug(
            f"Metric {metric.identifier} created for algorithm {algorithm_name}")
        
        self.acc[algorithm_name][metric.identifier] = metric

    @property
    def micro_metrics(self):
        results = defaultdict()
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}", metric.name)] = metric.micro_result
        return results

    @property
    def macro_metrics(self) -> defaultdict:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}")][metric.name] = metric.macro_result
        return results
    
    def df_micro_metric(self) -> pd.DataFrame:
        """Micro metric across all timestamps
        
        Computation of metrics evaluated on the user level

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.micro_metrics, orient="index").explode(["user_id","score"])
        df = df.rename_axis(["Algorithm", "Timestamp", "Metric"])
        return df
    
    def df_macro_metric(self) -> pd.DataFrame:
        """Macro metric across all timestamps

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = pd.DataFrame.from_dict(self.macro_metrics, orient="index")
        df = df.rename_axis(["Algorithm", "Timestamp"])
        return df
    
    def df_metric(self,
                  level:Union[Literal["micro","macro"], MetricLevelEnum]=MetricLevelEnum.MICRO,
                  filter_timestamp:Optional[int]=None,
                  filter_algo:Optional[str]=None
                  ) -> pd.DataFrame:
        if level == MetricLevelEnum.MICRO:
            df = self.df_micro_metric()
        else:
            df = self.df_macro_metric()
            
        if filter_algo:
            df = df.filter(like=filter_algo, axis=0)
        if filter_timestamp:
            df = df.filter(like=f"t={filter_timestamp}", axis=0)

        return df
    
    def df_global_metric(self) -> pd.DataFrame:
        """Global metric across all timestamps

        :return: _description_
        :rtype: pd.DataFrame
        """
        df = self.df_macro_metric().groupby("Algorithm").mean()
        return df

    # @property
    # def metrics_by_algo_time(self) -> defaultdict[Tuple[str], dict[Tuple[str], float]]:
    #     results = defaultdict(dict)
    #     for algo_name in self.acc:
    #         for metric_identifier in self.acc[algo_name]:
    #             metric = self.acc[algo_name][metric_identifier]
    #             results[(algo_name, f"t={metric.timestamp_limit}")][metric.name] = metric.macro_result
    #     return results

    # @property
    # def metrics_time_test(self) -> defaultdict[Tuple[str], dict[Tuple[str], float]]:
    #     results = defaultdict(dict)
    #     for algo_name in self.acc:
    #         for metric_identifier in self.acc[algo_name]:
    #             metric = self.acc[algo_name][metric_identifier]
    #             results[(algo_name, f"t={metric.timestamp_limit}")][metric.name] = metric.macro_result
    #     return results
    
    # def _compute_global_metrics(self) -> pd.DataFrame:
    #     """Compute the global metrics using simple weighted average.

    #     :return: _description_
    #     :rtype: _type_
    #     """
    #     tmp = self.df_by_algo_time()
    #     tmp = tmp.reset_index().drop("level_1",axis=1).rename(columns={"level_0":"Algo"})
    #     return tmp.groupby(['Algo']).mean()

    # @property
    # def global_metrics(self) -> pd.DataFrame:
    #     return self._compute_global_metrics()