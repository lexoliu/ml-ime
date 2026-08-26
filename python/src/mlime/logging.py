"""Shared structlog configuration."""

from __future__ import annotations

import logging

import structlog


def configure(verbose: bool = False) -> None:
    """Install a human-readable structlog renderer on stderr."""
    logging.basicConfig(format="%(message)s", level=logging.DEBUG if verbose else logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


log = structlog.get_logger()
