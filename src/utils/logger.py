"""
Structured Logging utility for recording application execution,
dialogue decisions, retrieval metrics, and latency.
"""

import json
import logging
import sys
import time
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """
    Format log entries as JSON strings for machine-readability and structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "line": record.lineno
        }
        
        # Add custom structured attributes passed in extra
        if hasattr(record, "structured_data"):
            log_data.update(record.structured_data)
            
        return json.dumps(log_data)


def setup_logger(name: str = "shl_recommender") -> logging.Logger:
    """
    Initializes a structured logger writing to standard out.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JSONFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger


# Global structured logger
app_logger = setup_logger()


def log_dialogue_event(
    message: str,
    turn: int,
    intent: str,
    confidence: float,
    latency_ms: float,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Logs dialog state transition events.
    """
    structured_data = {
        "event_type": "dialogue_decision",
        "turn": turn,
        "intent": intent,
        "confidence": confidence,
        "latency_ms": latency_ms,
        **(extra_data or {})
    }
    app_logger.info(message, extra={"structured_data": structured_data})


def log_retrieval_event(
    message: str,
    query: str,
    retrieved_count: int,
    recall_at_10: Optional[float] = None,
    extra_data: Optional[Dict[str, Any]] = None
) -> None:
    """
    Logs retrieval operations and pipeline metrics.
    """
    structured_data = {
        "event_type": "retrieval_query",
        "query": query,
        "retrieved_count": retrieved_count,
        "recall_at_10": recall_at_10,
        **(extra_data or {})
    }
    app_logger.info(message, extra={"structured_data": structured_data})
