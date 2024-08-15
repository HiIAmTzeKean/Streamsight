import logging
import random
import uuid
from typing import List, Optional
from warnings import warn

from scipy.sparse import csr_matrix

from streamsight.algorithms import Algorithm
from streamsight.evaluator.accumulator import (MacroMetricAccumulator,
                                               MicroMetricAccumulator)
from streamsight.evaluator.base import EvaluatorBase
from streamsight.evaluator.util import AlgorithmStatusWarning
from streamsight.matrix import InteractionMatrix
from streamsight.metrics import Metric
from streamsight.registries import METRIC_REGISTRY, MetricEntry
from streamsight.registries.registry import (AlgorithmStatusEntry,
                                             AlgorithmStatusEnum,
                                             AlgorithmStatusRegistry)
from streamsight.settings import Setting

logger = logging.getLogger(__name__)

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

        self.has_started = False
        self._next_window_ready = False
        self.seed = 42

    def start_stream(self):
        # TODO allow programmer to register anytime ?
        self.has_started = True
        self.setting.reset_data_generators()
        
        self._micro_acc = MicroMetricAccumulator()
        
        self._macro_acc = MacroMetricAccumulator()
        for algo in self.status_registry:
            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                if metric_entry.K is not None:
                    metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=None,cache=True)
                else:
                    metric:Metric = metric_cls(timestamp_limit=None,cache=True)
                self._macro_acc.add(metric=metric, algorithm_name=algo)
        
        background_data = self.setting.background_data
        self.user_item_base._update_known_user_item_base(background_data)
        background_data.mask_shape(self.user_item_base.known_shape)
        self._training_data_cache = background_data
        
        self._cache_evaluation_data()

    def register_algorithm(self, algorithm: Algorithm) -> str:
        if self.has_started:
            raise ValueError("Cannot register algorithm after the stream has started")
    
        # assign a unique identifier to the algorithm
        rd = random.Random(x=self.seed)
        algo_id = uuid.UUID(int=rd.getrandbits(128), version=4)
        # store the algorithm in the registry
        self.status_registry[algo_id] = AlgorithmStatusEntry(
            name=algorithm.identifier,
            algo_id=algo_id,
            status=AlgorithmStatusEnum.NEW,
            algo_ptr=algorithm,
        )
        return algo_id

    def get_data(self, algo_id: str) -> InteractionMatrix:
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
        status = self.status_registry[algo_id].status
        
        if status == AlgorithmStatusEnum.COMPLETED:
            warn(AlgorithmStatusWarning(algo_id, status, "data_release"))

        elif status in [AlgorithmStatusEnum.PREDICTED, AlgorithmStatusEnum.NEW]:
            self.status_registry.update(algo_id, AlgorithmStatusEnum.READY)

        # release data to the algorithm
        return self._training_data_cache

    def get_unlabeled_data(self, algo_id: str) -> Optional[InteractionMatrix]:
        status = self.status_registry[algo_id].status
        if status in [AlgorithmStatusEnum.READY, AlgorithmStatusEnum.PREDICTED]:
            return self._unlabeled_data_cache
        
        if status == AlgorithmStatusEnum.COMPLETED:
            warn(AlgorithmStatusWarning(algo_id, status, "complete"))
            return
        warn(AlgorithmStatusWarning(algo_id, status, "unlabeled"))
    def submit_prediction(self, algo_id: str, X_pred: csr_matrix) -> None:
        status = self.status_registry[algo_id].status
        
        if status == AlgorithmStatusEnum.READY:
            self._evaluate(algo_id, X_pred)
            self.status_registry.update(algo_id, AlgorithmStatusEnum.PREDICTED)
        
        elif status == AlgorithmStatusEnum.PREDICTED:
            warn(AlgorithmStatusWarning(algo_id, status, "predict_complete"))
        
        else:
            warn(AlgorithmStatusWarning(algo_id, status, "predict"))
        
    def _cache_evaluation_data(self):
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
    
    def _evaluate(self, algo_id: str, X_pred: csr_matrix):
        X_true = self._ground_truth_data_cache.binary_values

        # evaluate the prediction
        for metric_entry in self.metric_entries:
            metric_cls = METRIC_REGISTRY.get(metric_entry.name)
            if metric_entry.K is not None:
                metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=self._timestamp_data_cache,cache=True)
            else:
                metric:Metric = metric_cls(timestamp_limit=self._timestamp_data_cache,cache=True)
            metric.calculate(X_true, X_pred)
            self._micro_acc.add(metric=metric, algorithm_name=algo_id)
        
        self._macro_acc.cache_results(algo_id, X_true, X_pred)
