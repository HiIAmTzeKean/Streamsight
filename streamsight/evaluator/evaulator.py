import logging
from typing import List, Set
from streamsight.algorithms.base import Algorithm
from streamsight.matrix.interaction_matrix import InteractionMatrix
from streamsight.metrics.base import Metric
from streamsight.pipelines.pipeline import MetricAccumulator
from streamsight.registries.registry import ALGORITHM_REGISTRY, METRIC_REGISTRY, AlgorithmEntry, MetricEntry
from streamsight.setting.base_setting import Setting
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Evaluator(object):
    def __init__(self,
                 algorithm_entries: List[AlgorithmEntry],
                 metric_entries: List[MetricEntry],
                 setting: Setting,
                 ignore_unknown_user: bool = True,
                 ignore_unknown_item: bool = True):
        self.algorithm_entries = algorithm_entries
        self.metric_entries = metric_entries
        
        self.setting = setting
        self.algorithm: List[Algorithm] = []
        self.metric_acc = MetricAccumulator()
        """Accumulator for computed metrics"""
        self.ignore_unknown_user = ignore_unknown_user
        self.ignore_unknown_item = ignore_unknown_item
        
        self.unknown_user = set()
        self.known_user = set()
        self.unknown_item = set()
        self.known_item = set()
    
    def _update_known_user_item_base(self, data:InteractionMatrix):
        self.known_item.update(data.item_ids)
        self.known_user.update(data.user_ids)
        
    def _update_unknown_user_item_base(self, data:InteractionMatrix):
        self.unknown_user = data.user_ids.difference(self.known_user)
        self.unknown_item = data.item_ids.difference(self.known_item)

    def _reset_unknown_user_item_base(self):
        self.unknown_user = set()
        self.unknown_item = set()
        
    def _instantiate_algorithm(self):
         for algorithm_entry in self.algorithm_entries:
            params = algorithm_entry.params
            self.algorithm.append(ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params))
            
    def run(self):
        self._instantiate_algorithm()

        background_data = self.setting.background_data
        self._update_known_user_item_base(background_data)
        background_data.mask_shape((len(self.known_user), len(self.known_item)))
        
        for algo in self.algorithm:
            algo.fit(background_data)
            
        for _ in tqdm(range(self.setting.num_split)):
            # we assume that we ignore all unknown user and item now
            unlabeled_data = self.setting.next_unlabeled_data()
            # unlabeled data will respect the unknown user and item
            # and thus will take the shape of the known user and item
            ground_truth_data = self.setting.next_ground_truth_data()
            # the ground truth must follow the same shape as the unlabeled data
            # for evaluation purposes. This means that we drop the unknown user and item
            # from the ground truth data
            
            incremental_data = self.setting.next_incremental_data()
            current_timestamp = self.setting.next_data_timestamp_limit()
            
            self._update_unknown_user_item_base(ground_truth_data)
            # Assume that we ignore unknowns
            unlabeled_data.mask_shape((len(self.known_user), len(self.known_item)))
            
            #TODO based on the unknown flags, remove unknown users
            ground_truth_data.users_not_in(self.unknown_user,inplace=True)
            ground_truth_data.items_not_in(self.unknown_item,inplace=True)
            ground_truth_data.mask_shape((len(self.known_user), len(self.known_item)))
            
            self._reset_unknown_user_item_base()
            
            
            for algo in self.algorithm:
                X_pred = algo.predict(unlabeled_data)
                
                for metric_entry in self.metric_entries:
                    metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                    if metric_entry.K is not None:
                        metric = metric_cls(K=metric_entry.K, timestamp_limit=current_timestamp)
                    else:
                        metric = metric_cls(timestamp_limit=current_timestamp)
                    metric.calculate(ground_truth_data.binary_values, X_pred)
                    self.metric_acc.add(metric=metric, algorithm_name=algo.identifier)
                
            
            self._update_known_user_item_base(incremental_data)
            incremental_data.mask_shape((len(self.known_user), len(self.known_item)))
            
            for algo in self.algorithm:
                algo.fit(incremental_data)