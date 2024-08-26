"""
.. currentmodule:: streamsight.settings

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

.. autosummary::
    :toctree: generated/

    Setting
    SingleTimePointSetting
    SlidingWindowSetting

A setting is stateful. Thus, the initialization of the setting object only stores
the parameters that are passed. Calling of :attr:`Setting.split` is necessary
such that the attributes :attr:`Setting.background_data`, :attr:`Setting.unlabeled_data`
and :attr:`Setting.ground_truth_data` are populated.

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

.. autosummary::
    :toctree: generated/

    TimestampSplitter
    NPastInteractionTimestampSplitter

Processor
------------
The processor module contains classes that are used to process the data.


.. autosummary::
    :toctree: generated/

    Processor
    PredictionDataProcessor
    
Exception
------------
The exception class :class:`EOWSetting` is used to raise an exception when the
end of window is reached. Note that this exception is declared the base file.

.. autosummary::
    :toctree: generated/

    EOWSetting
"""

from streamsight.settings.base import Setting, EOWSetting
from streamsight.settings.single_time_point_setting import (
    SingleTimePointSetting,
)
from streamsight.settings.sliding_window_setting import SlidingWindowSetting
from streamsight.settings.leave_n_out_setting import LeaveNOutSetting
from streamsight.settings.processor import Processor, PredictionDataProcessor
from streamsight.settings.splitters import (
    TimestampSplitter,
    NPastInteractionTimestampSplitter,
    NLastInteractionSplitter
)
