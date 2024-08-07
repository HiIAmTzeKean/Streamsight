"""
Setting
------------

The setting module contains classes that define how the data is split. The
specifics of the split are defined in paper referenced. A setting will contain
the following components:

- background_data: Data that is used to train the algorithm before the first\
    split.
    
- unlabeled_data: Data that is released to the algorithm for prediction. Contains\
    the last n interactions split. The purpose is to provide sequential data\
    to the algorithm. The data also contains the ID to be predicted and is labeled\
    with "-1". Timestamps of the interactions to be predicted are preserved.
    
- ground_truth_data: Data that is used to evaluate the algorithm. This data\
    will contain the actual interactions.

.. currentmodule:: streamsight.setting

.. autosummary::
    :toctree: generated/

    Setting
    SingleTimePointSetting
    SlidingWindowSetting

A setting is stateful. Thus, the initialisation of the setting object only stores
the parameters that are passed. Calling of :attr:`Scenario.split` is necessary
such that the attributes :attr:`Scenario.background_data`, :attr:`Scenario.unlabeled_data`
and :attr:`Scenario.ground_truth_data` are populated.

Splitters
------------

Splitters are stateless and can be reused across different settings. The role of
the splitter is to split the data into two parts. That is :attr:`past_interaction`
and :attr:`future_interaction`. The naming is chosen such that data returned is
easily understood as dataset being dealt with will always contain timestamps as
in our assumption for this project.

Implementation of other settings is possible by implementing new splitters to
define the split such that there is no need to redefine the entire architecture
of the setting.

For eg. the sliding window setting can be implemented by using a splitter that
may split using a different or additional criteria other than timestamp. The
programmer instead of redefining the entire setting can just implement a new
splitter and reuse the existing setting.

.. currentmodule:: streamsight.setting.splitters

.. autosummary::
    :toctree: generated/

    TimestampSplitter
    NPastInteractionTimestampSplitter

Processor
------------

.. currentmodule:: streamsight.setting.processor

.. autosummary::
    :toctree: generated/

    Processor
    PredictionDataProcessor
"""


from streamsight.setting.base_setting import Setting
from streamsight.setting.single_time_point_setting import SingleTimePointSetting
from streamsight.setting.sliding_window_setting import SlidingWindowSetting
from streamsight.setting.processor import Processor, PredictionDataProcessor
from streamsight.setting.splitters import TimestampSplitter, NPastInteractionTimestampSplitter