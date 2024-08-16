import logging
import random
import uuid
from uuid import UUID
from typing import Dict, List, Literal, Optional, Union
from warnings import warn
import warnings

import pandas as pd

from scipy.sparse import csr_matrix

from streamsight.algorithms import Algorithm
from streamsight.evaluator.accumulator import (MacroMetricAccumulator,
                                               MicroMetricAccumulator)
from streamsight.evaluator.base import EvaluatorBase
from streamsight.evaluator.util import AlgorithmStatusWarning, MetricLevelEnum
from streamsight.matrix import InteractionMatrix
from streamsight.metrics import Metric
from streamsight.registries import METRIC_REGISTRY, MetricEntry
from streamsight.registries.registry import (AlgorithmStatusEntry,
                                             AlgorithmStateEnum,
                                             AlgorithmStatusRegistry)
from streamsight.settings import Setting

logger = logging.getLogger(__name__)
warnings.simplefilter('always', AlgorithmStatusWarning)

class EvaluatorStreamer(EvaluatorBase):
    def __init__(
        self,
        metric_entries: List[MetricEntry],
        setting: Setting,
        ignore_unknown_user: bool = True,
        ignore_unknown_item: bool = True,
    ):
        super().__init__(metric_entries, setting, ignore_unknown_user, ignore_unknown_item)
        
        self.status_registry = AlgorithmStatusRegistry()
        
        self._unlabeled_data_cache: InteractionMatrix
        self._ground_truth_data_cache: InteractionMatrix
        self._training_data_cache: InteractionMatrix
        self._timestamp_data_cache: int

        self.has_started = False
        
        self.seed = 42
        self.rd = random.Random(self.seed)

    def start_stream(self):
        # TODO allow programmer to register anytime ?
        self.has_started = True
        self.setting.reset_data_generators()
        
        logger.debug(f"Preparing ")
        self._micro_acc = MicroMetricAccumulator()
        
        self._macro_acc = MacroMetricAccumulator()
        for algo_id in self.status_registry:
            algo_name = self.status_registry.get_algorithm_identifier(algo_id)
            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                if metric_entry.K is not None:
                    metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=None,cache=True)
                else:
                    metric:Metric = metric_cls(timestamp_limit=None,cache=True)
                self._macro_acc.add(metric=metric, algorithm_name=algo_name)
        
        background_data = self.setting.background_data
        self.user_item_base._update_known_user_item_base(background_data)
        background_data.mask_shape(self.user_item_base.known_shape)
        self._training_data_cache = background_data
        
        self._cache_evaluation_data()

    def metric_results(self,
                       level:Union[Literal["micro","macro"], MetricLevelEnum]="macro",
                       only_current_frame=False,
                       filter_algo:Optional[str]=None) -> pd.DataFrame:
        """Results of the metrics computed.
        
        This property returns the results of the metrics computed in the evaluator.
        """
        timestamp = None
        if only_current_frame:
            timestamp = self._timestamp_data_cache

        acc = self._macro_acc
        if level == MetricLevelEnum.MICRO:
            acc = self._micro_acc
        
        return acc.df_metric(filter_algo=filter_algo,
                             filter_timestamp=timestamp)

    def register_algorithm(self, algorithm: Algorithm) -> UUID:
        if self.has_started:
            raise ValueError("Cannot register algorithm after the stream has started")
    
        # assign a unique identifier to the algorithm
        algo_id = uuid.UUID(int=self.rd.getrandbits(128), version=4)
        logger.info(f"Registering algorithm {algorithm.identifier} with ID: {algo_id}")
        
        # store the algorithm in the registry
        self.status_registry[algo_id] = AlgorithmStatusEntry(
            name=algorithm.identifier,
            algo_id=algo_id,
            state=AlgorithmStateEnum.NEW,
            algo_ptr=algorithm,
        )
        logger.debug(f"Algorithm {algo_id} registered")
        return algo_id
    
    def get_algorithm_state(self, algo_id: UUID) -> AlgorithmStateEnum:
        return self.status_registry[algo_id].state
    
    def get_all_algorithm_status(self) -> Dict[str, AlgorithmStateEnum]:
        return self.status_registry.all_algo_states()

    def get_data(self, algo_id: UUID) -> InteractionMatrix:
        if not self.has_started:
            raise ValueError(f"call start_stream() before requesting data for algorithm {algo_id}")
        
        # check if we need to move to the next window
        if self.setting.is_sliding_window_setting and self.status_registry.is_all_predicted():
            self.user_item_base._reset_unknown_user_item_base()
            incremental_data = self.setting.next_incremental_data()
            self.user_item_base._update_known_user_item_base(incremental_data)
            incremental_data.mask_shape(self.user_item_base.known_shape)
            self._training_data_cache = incremental_data
            
            self._cache_evaluation_data()
            
        # check status of of algo
        status = self.status_registry[algo_id].state
        
        if status == AlgorithmStateEnum.COMPLETED:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))

        elif status == AlgorithmStateEnum.NEW:
            self.status_registry.update(algo_id, AlgorithmStateEnum.READY, self._timestamp_data_cache)
            
        elif status == AlgorithmStateEnum.READY and self.status_registry[algo_id].data_segment == self._timestamp_data_cache:
            warn(AlgorithmStatusWarning(algo_id, status, "data_release"))
            
        elif status == AlgorithmStateEnum.PREDICTED and self.status_registry[algo_id].data_segment == self._timestamp_data_cache:
            # if algo has predicted, check if current timestamp has not changed
            return_msg = f"Algorithm {algo_id} has already predicted for this data segment, please wait for all other algorithms to predict"
            logger.info(return_msg)
            print(return_msg)
            
        else:            
            self.status_registry.update(algo_id, AlgorithmStateEnum.READY, self._timestamp_data_cache)
            
        # release data to the algorithm
        return self._training_data_cache

    def get_unlabeled_data(self, algo_id: UUID) -> Optional[InteractionMatrix]:
        status = self.status_registry[algo_id].state
        if status in [AlgorithmStateEnum.READY, AlgorithmStateEnum.PREDICTED]:
            return self._unlabeled_data_cache
        
        if status == AlgorithmStateEnum.COMPLETED:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))
            return
        warn(AlgorithmStatusWarning(algo_id, status, "unlabeled"))
        
    def submit_prediction(self, algo_id: UUID, X_pred: csr_matrix) -> None:
        status = self.status_registry[algo_id].state
        
        if status == AlgorithmStateEnum.READY:
            self._evaluate(algo_id, X_pred)
            self.status_registry.update(algo_id, AlgorithmStateEnum.PREDICTED)
            
        elif status == AlgorithmStateEnum.NEW:
            warn(AlgorithmStatusWarning(algo_id, status, "predict"))
        
        elif status == AlgorithmStateEnum.PREDICTED:
            warn(AlgorithmStatusWarning(algo_id, status, "predict_complete"))
        
        else:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))
            return
        
        if self._run_step == self.setting.num_split:
            self.status_registry.update(algo_id, AlgorithmStateEnum.COMPLETED)
            logger.info(f"Finished streaming")
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))
        
    def _cache_evaluation_data(self):
        self._run_step += 1
        
        unlabeled_data = self.setting.next_unlabeled_data()
        ground_truth_data = self.setting.next_ground_truth_data()
        self._timestamp_data_cache = self.setting.next_data_timestamp_limit()
        
        self.user_item_base._update_unknown_user_item_base(ground_truth_data)
        # Assume that we ignore unknowns
        unlabeled_data.mask_shape(self.user_item_base.known_shape)
        ground_truth_data.mask_shape(self.user_item_base.known_shape,
                                        drop_unknown_user=self.ignore_unknown_user,
                                        drop_unknown_item=self.ignore_unknown_item)
        
        self._unlabeled_data_cache = unlabeled_data
        self._ground_truth_data_cache = ground_truth_data
    
    def _evaluate(self, algo_id: UUID, X_pred: csr_matrix):
        X_true = self._ground_truth_data_cache.binary_values
        algorithm_name = self.status_registry.get_algorithm_identifier(algo_id)
        # evaluate the prediction
        for metric_entry in self.metric_entries:
            metric_cls = METRIC_REGISTRY.get(metric_entry.name)
            if metric_entry.K is not None:
                metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=self._timestamp_data_cache)
            else:
                metric:Metric = metric_cls(timestamp_limit=self._timestamp_data_cache)
            metric.calculate(X_true, X_pred)
            self._micro_acc.add(metric=metric, algorithm_name=algorithm_name)
        
        self._macro_acc.cache_results(algorithm_name, X_true, X_pred)
