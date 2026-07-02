"""
Constants module defining core business limits, schema definitions,
and state values for the SHL Assessment Recommender.
"""

from typing import Final

# Turn configuration for conversation budget enforcement
MAX_CONVERSATION_TURNS: Final[int] = 8
TURNS_WARNING_THRESHOLD: Final[int] = 6
RECOMMENDATION_FORCE_TURN: Final[int] = 7

# Timeout configuration in seconds for external service API calls
API_TIMEOUT_SECONDS: Final[float] = 25.0

# Catalog filter options and mapped types
class TestTypes:
    COGNITIVE: Final[str] = "K"
    PERSONALITY: Final[str] = "P"
    SKILLS: Final[str] = "S"
    LANGUAGE: Final[str] = "L"

# Retrieval boundary defaults
DEFAULT_RECALL_K: Final[int] = 10
MIN_RECOMMENDATIONS: Final[int] = 1
MAX_RECOMMENDATIONS: Final[int] = 10

# Confidence decision engine thresholds
CONFIDENCE_THRESHOLD_CLARIFY: Final[float] = 0.45
CONFIDENCE_THRESHOLD_RECOMMEND: Final[float] = 0.70

# Static error messages and standard responses
ERROR_UNEXPECTED: Final[str] = "An unexpected error occurred. Please try again."
REFUSAL_OFF_TOPIC: Final[str] = (
    "I can only help you with queries related to SHL assessment recommendations. "
    "For other general hiring or legal inquiries, please consult SHL's official support."
)
REFUSAL_SAFETY: Final[str] = (
    "Your query cannot be processed because it violates safety and topic boundaries."
)
