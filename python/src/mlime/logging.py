"""Shared structlog configuration.

structlog prints straight to the stream rather than through the standard library,
so the two levels here are independent: ``structlog``'s controls this pipeline's
own diagnostics, and the standard library's controls everything the dependencies
emit. The dependencies log one INFO line per HTTP request, which at the volumes
this pipeline works at *is* the log, so they are held at WARNING until ``--verbose``
asks for them. That is a level, not a list of library names, so a new dependency
cannot quietly reintroduce the noise.
"""

from __future__ import annotations

import logging

import structlog


def configure(verbose: bool = False) -> None:
    """Install a human-readable structlog renderer on stderr."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(message)s", level=logging.DEBUG if verbose else logging.WARNING)
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%H:%M:%S"),
            structlog.dev.ConsoleRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )


log = structlog.get_logger()
