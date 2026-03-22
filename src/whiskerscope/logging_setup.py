from __future__ import annotations

import logging
import logging.config
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whiskerscope.config import WhiskerscopeConfig

correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = correlation_id.get()  # type: ignore[attr-defined]
        return True


def setup_logging(config: WhiskerscopeConfig) -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "correlation": {"()": CorrelationFilter},
            },
            "formatters": {
                "structured": {
                    "format": (
                        '{"ts":"%(asctime)s","level":"%(levelname)s",'
                        '"logger":"%(name)s","cid":"%(correlation_id)s",'
                        '"msg":"%(message)s"}'
                    ),
                    "datefmt": "%Y-%m-%dT%H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "structured",
                    "filters": ["correlation"],
                },
            },
            "root": {
                "level": config.log_level,
                "handlers": ["console"],
            },
        }
    )
