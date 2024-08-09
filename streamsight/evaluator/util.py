import logging
from enum import StrEnum

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
    
    
