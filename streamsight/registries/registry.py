from typing import Any, Dict, NamedTuple, Optional

import streamsight.algorithms
import streamsight.datasets
import streamsight.metrics


class Registry:
    """
    A Registry is a wrapper for a dictionary that maps
    names to Python types (most often classes).
    """

    def __init__(self, src):
        self.registered: Dict[str, type] = {}
        self.src = src

    def __getitem__(self, key: str) -> type:
        """Retrieve the type for the given key.

        :param key: the key of the type to fetch
        :type key: str
        :returns: The class type associated with the key
        :rtype: type
        """
        return self.get(key)

    def __contains__(self, key: str) -> bool:
        """Check if the given key is known to the registry.

        :param key: The key to check.
        :type key: str
        :return: True if the key is known
        :rtype: bool
        """
        try:
            self.get(key)
            return True
        except AttributeError:
            return False

    def get(self, key: str) -> type:
        """Retrieve the value for this key. This value is a Python type (most often a class).

        :param key: The key to fetch
        :type key: str
        :return: The class type associated with the key
        :rtype: type
        """
        if key in self.registered:
            return self.registered[key]
        else:
            return getattr(self.src, key)

    def register(self, key: str, c: type):
        """Register a new Python type (most often a class).

        After registration, the key can be used to fetch the Python type from the registry.

        :param key: key to register the type at. Needs to be unique to the registry.
        :type key: str
        :param c: class to register.
        :type c: type
        """
        if key in self:
            raise KeyError(f"key {key} already registered")
        self.registered[key] = c
        
class AlgorithmRegistry(Registry):
    """Registry for easy retrieval of algorithm types by name.

    The registry comes preregistered with all streamsight algorithms.
    """

    def __init__(self):
        super().__init__(streamsight.algorithms)

class MetricRegistry(Registry):
    """Registry for easy retrieval of metric types by name.

    The registry comes preregistered with all the streamsight metrics.
    """

    def __init__(self):
        super().__init__(streamsight.metrics)

class DatasetRegistry(Registry):
    """Registry for easy retrieval of dataset types by name.

    The registry comes preregistered with all the streamsight datasets.
    """

    def __init__(self):
        super().__init__(streamsight.datasets)
    

ALGORITHM_REGISTRY = AlgorithmRegistry()
"""Registry for algorithms.

Contains the streamsight algorithms by default,
and allows registration of new algorithms via the `register` function.

Example::

    from streamsight.pipelines import ALGORITHM_REGISTRY

    # Construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get('ItemKNN')(K=20)

    from streamsight.algorithms import ItemKNN
    ALGORITHM_REGISTRY.register('HelloWorld', ItemKNN)

    # Also construct an ItemKNN object with parameter K=20
    algo = ALGORITHM_REGISTRY.get('HelloWorld')(K=20)
"""

METRIC_REGISTRY = MetricRegistry()
"""Registry for metrics.

Contains the streamsight metrics by default,
and allows registration of new metrics via the `register` function.

Example::

    from streamsight.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from streamsight.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)

"""


DATASET_REGISTRY = DatasetRegistry()
"""Registry for datasets.

Contains the streamsight metrics by default,
and allows registration of new metrics via the `register` function.

Example::

    from streamsight.pipelines import METRIC_REGISTRY

    # Construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('Recall')(K=20)

    from streamsight.algorithms import Recall
    METRIC_REGISTRY.register('HelloWorld', Recall)

    # Also construct a Recall object with parameter K=20
    algo = METRIC_REGISTRY.get('HelloWorld')(K=20)
"""


class AlgorithmEntry(NamedTuple):
    """Config class to represent an algorithm when configuring the pipeline.

    :param name: Name of the algorithm
    :type name: str
    :param params: Parameters that do not require optimization as key-value pairs,
        where the key is the name of the hyperparameter and value is the value it should take.
    :type params: Dict[str, Any], optional
    """

    name: str
    params: Optional[Dict[str, Any]] = None


class MetricEntry(NamedTuple):
    """Config class to represent an algorithm when configuring the pipeline.

    :param name: Name of the algorithm
    :type name: str
    :param K: Top K value for the metric.
    :type K: int, optional
    """
    name: str
    K: Optional[int] = None