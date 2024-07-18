from collections import defaultdict
import logging
import os
from typing import Tuple, List, Optional
from warnings import warn

import pandas as pd
from scipy.sparse import csr_matrix
from streamsight.metrics.base import Metric
from tqdm.auto import tqdm

from streamsight.algorithms.base import Algorithm
from streamsight.matrix import InteractionMatrix
from streamsight.registries import (
    ALGORITHM_REGISTRY,
    METRIC_REGISTRY,
    AlgorithmEntry,
    MetricEntry,
)

logger = logging.getLogger(__name__)


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

    # ? should this be cached
    @property
    def metrics(self):
        results = defaultdict(dict)
        for key in self.acc:
            for k in self.acc[key]:
                results[key][k] = self.acc[key][k].value
        return results

    def df_by_algo(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.metrics).T

    def df_by_metrics(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.metrics)
    
    def df_by_time_algo(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.metrics_by_algo_time, orient="index").swaplevel(0,1).sort_index(level=0)
    
    def df_by_algo_time(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.metrics_by_algo_time, orient="index")
    
    @property
    def metrics_by_time(self) -> defaultdict[str, defaultdict[str, dict[str, float]]]:
        results = defaultdict(lambda: defaultdict(dict))
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[algo_name][f"t={metric.timestamp_limit}"][metric.name] = metric.value
        return results

    @property
    def metrics_by_algo_time(self) -> defaultdict[Tuple[str], dict[Tuple[str], float]]:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(algo_name, f"t={metric.timestamp_limit}")][metric.name] = metric.value
        return results

    @property
    def metrics_by_time_algo(self) -> defaultdict[Tuple[str], dict[Tuple[str], float]]:
        results = defaultdict(dict)
        for algo_name in self.acc:
            for metric_identifier in self.acc[algo_name]:
                metric = self.acc[algo_name][metric_identifier]
                results[(f"t={metric.timestamp_limit}", algo_name)][metric.name] = metric.value
        return results
    
    def _compute_global_metrics(self) -> pd.DataFrame:
        """Compute the global metrics using simple weighted average.

        :return: _description_
        :rtype: _type_
        """
        tmp = self.df_by_algo_time()
        tmp = tmp.reset_index().drop("level_1",axis=1).rename(columns={"level_0":"Algo"})
        return tmp.groupby(['Algo']).mean()

    @property
    def global_metrics(self) -> pd.DataFrame:
        return self._compute_global_metrics()

