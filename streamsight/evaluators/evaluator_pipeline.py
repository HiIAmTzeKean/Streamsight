import logging
import warnings
from typing import List
from warnings import warn

from tqdm import tqdm

from streamsight.algorithms import Algorithm
from streamsight.evaluators.accumulator import (MacroMetricAccumulator,
                                                MicroMetricAccumulator)
from streamsight.evaluators.base import EvaluatorBase
from streamsight.metrics import Metric
from streamsight.registries import (ALGORITHM_REGISTRY, METRIC_REGISTRY,
                                    AlgorithmEntry, MetricEntry)
from streamsight.settings import Setting

logger = logging.getLogger(__name__)

class EvaluatorPipeline(EvaluatorBase):
    """Evaluation via pipeline
    
    The pipeline is responsible for evaluating algorithms with metrics and evaluates
    the algorithms in 3 phases:
    
    1. Pre-phase
    2. Evaluation phase
    3. Data release phase
    
    In the classical split setting, the evaluator will only run phase 1 and 2.
    In the sliding window setting, the evaluator will run all 3 phases, with
    phase 2 and 3 repeated for each split.
    """
    def __init__(self,
                 algorithm_entries: List[AlgorithmEntry],
                 metric_entries: List[MetricEntry],
                 setting: Setting,
                 ignore_unknown_user: bool = True,
                 ignore_unknown_item: bool = True):
        super().__init__(metric_entries, setting, ignore_unknown_user, ignore_unknown_item)
        
        self.algorithm_entries = algorithm_entries
        self.algorithm: List[Algorithm]

        # internal state
        self._current_timestamp: int
            
    def _instantiate_algorithm(self):
        """Instantiate the algorithms from the algorithm entries.
        
        This method instantiates the algorithms and stores them in :attr:`algorithm`.
        Each time this method is called, the algorithms are re-instantiated.
        """
        self.algorithm = []
        for algorithm_entry in self.algorithm_entries:
            params = algorithm_entry.params
            self.algorithm.append(ALGORITHM_REGISTRY.get(algorithm_entry.name)(**params))
            
    def _ready_algo(self):
        """Train the algorithms with the background data.
        
        This method should be called after :meth:`_instantiate_algorithm()`. The
        algorithms are trained with the background data, and the set of known
        user/item is updated.

        :raises ValueError: If algorithm is not instantiated
        """
        if not hasattr(self, "algorithm"):
            raise ValueError("Algorithm not instantiated")
        background_data = self.setting.background_data
        self.user_item_base._update_known_user_item_base(background_data)
        background_data.mask_shape(self.user_item_base.known_shape)
        
        for algo in self.algorithm:
            algo.fit(background_data)
    
    def _ready_evaluator(self):
        """Pre-phase of the evaluator. (Phase 1)
        
        This method prepares the evaluator for the evaluation process.
        It instantiates the algorithm, trains the algorithm with the background data,
        instantiates the metric accumulator, and prepares the data generators.
        The next phase of the evaluator is the evaluation phase.
        """
        logger.info("Phase 1: Preparing the evaluator...")
        self._instantiate_algorithm()
        self._ready_algo()
        logger.debug(f"Algorithms trained with background data...")
        
        self._micro_acc = MicroMetricAccumulator()
        
        self._macro_acc = MacroMetricAccumulator()
        for algo in self.algorithm:
            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                if metric_entry.K is not None:
                    metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=None,cache=True)
                else:
                    metric:Metric = metric_cls(timestamp_limit=None,cache=True)
                self._macro_acc.add(metric=metric, algorithm_name=algo.identifier)
        logger.debug(f"Metric accumulator instantiated...")
        
        self.setting.reset_data_generators()
        logger.debug(f"Setting data generators ready...")
    
    def _evaluate_step(self):
        """Evaluate performance of the algorithms. (Phase 2)
        
        This method evaluates the performance of the algorithms with the metrics.
        It takes the unlabeled data, predicts the interaction, and evaluates the
        performance with the ground truth data.
        """
        logger.info("Phase 2: Evaluating the algorithms...")
        # we assume that we ignore all unknown user and item now
        unlabeled_data = self.setting.next_unlabeled_data()
        # unlabeled data will respect the unknown user and item
        # and thus will take the shape of the known user and item
        ground_truth_data = self.setting.next_ground_truth_data()
        # the ground truth must follow the same shape as the unlabeled data
        # for evaluation purposes. This means that we drop the unknown user and item
        # from the ground truth data
        current_timestamp = self.setting.next_data_timestamp_limit()
        self._current_timestamp = current_timestamp
        
        self.user_item_base._update_unknown_user_item_base(ground_truth_data)
        # Assume that we ignore unknowns
        with warnings.catch_warnings(action="ignore"):
            unlabeled_data.mask_shape(self.user_item_base.known_shape)
        ground_truth_data.mask_shape(self.user_item_base.known_shape,
                                        drop_unknown_user=self.ignore_unknown_user,
                                        drop_unknown_item=self.ignore_unknown_item)
        
        X_true = ground_truth_data.binary_values
        for algo in self.algorithm:
            X_pred = algo.predict(unlabeled_data)
            
            if X_pred.shape != X_true.shape:
                # shapes might not be the same in the case of dropping unknowns
                # from the ground truth data. We ensure that the same unknowns
                # are dropped from the predictions
                X_pred = X_pred[:X_true.shape[0], :X_true.shape[1]]

            for metric_entry in self.metric_entries:
                metric_cls = METRIC_REGISTRY.get(metric_entry.name)
                if metric_entry.K is not None:
                    metric:Metric = metric_cls(K=metric_entry.K, timestamp_limit=current_timestamp)
                else:
                    metric:Metric = metric_cls(timestamp_limit=current_timestamp)
                metric.calculate(X_true, X_pred)
                self._micro_acc.add(metric=metric, algorithm_name=algo.identifier)
            
            # macro metric purposes
            self._macro_acc.cache_results(algo.identifier, X_true, X_pred)
        
        self.user_item_base._reset_unknown_user_item_base()
    
    def _data_release_step(self):
        """Data release phase. (Phase 3)
        """
        if not self.setting.is_sliding_window_setting:
            return
        logger.info("Phase 3: Releasing the data...")
        
        incremental_data = self.setting.next_incremental_data()
        self.user_item_base._update_known_user_item_base(incremental_data)
        incremental_data.mask_shape(self.user_item_base.known_shape)
        
        for algo in self.algorithm:
            algo.fit(incremental_data)
        
    def run_step(self, reset=False):
        """Run a single step of the evaluator.
        
        This method runs a single step of the evaluator. The evaluator is split
        into 3 phases. In the first run, all 3 phases are run. In the subsequent
        runs, only the evaluation and data release phase are run. The method
        will run all steps until the number of splits is reached. To rerun the
        evaluation again, call with `reset=True`.

        :param reset: To reset the evaluation step , defaults to False
        :type reset: bool, optional
        """
        if reset:
            self._run_step = 0
        
        self._run_step += 1
        if self._run_step == 1:
            logger.info(f"There is a total of {self.setting.num_split} steps."
                        f" Running step {self._run_step}")
            self._ready_evaluator()
            logger.info(f"Evaluator ready...")
            self._evaluate_step()
            self._data_release_step()
            return
        
        if self._run_step > self.setting.num_split:
            logger.info(f"Finished running all steps, call `run_step(reset=True)` to run the evaluation again")
            warn(f"Running this method again will not have any effect.")
            return
        logger.info(f"Running step {self._run_step}")
        self._evaluate_step()
        self._data_release_step()
    
    def run_steps(self, num_steps:int):
        """Run multiple steps of the evaluator.

        :param num_steps: Number of steps to run
        :type num_steps: int
        """
        for _ in tqdm(range(num_steps)):
            self.run_step()      

    def run(self):
        """Run the evaluator across all steps and splits
        
        Runs all 3 phases across all splits (if there are multiple splits).
        This method should be called when the programmer wants to step through
        all phases and splits to arrive to the metrics computed. An alternative
        to running through all splits is to call :meth:`run_step` method which runs
        only one step at a time.
        """
        self._ready_evaluator()
            
        for _ in tqdm(range(self.setting.num_split)):
            self._evaluate_step()
            self._data_release_step()
