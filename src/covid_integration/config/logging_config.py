"""
COVID-19 Data Integration - Centralized Logging Configuration

This module provides centralized logging configuration for the entire project.
Ensures consistent log formatting, levels, and output across all modules.
"""

import logging
import sys
from typing import Optional

from .constants import LOG_FORMAT, LOG_LEVEL

# Global flag to prevent duplicate configuration
_LOGGING_CONFIGURED = False


def configure_logging(
    level: str = LOG_LEVEL, format_string: str = LOG_FORMAT, suppress_external: bool = True
) -> None:
    """
    Configure logging for the entire application.

    Args:
        level: Global logging level
        format_string: Log message format
        suppress_external: Whether to suppress verbose external library logs
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED:
        return

    # Clear any existing handlers to prevent duplication
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Suppress verbose external library logging
    if suppress_external:
        external_loggers = [
            "urllib3.connectionpool",
            "requests.packages.urllib3",
            "matplotlib",
            "PIL",
            "plotly",
        ]

        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    _LOGGING_CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with standard configuration.

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Logger instance
    """
    # Ensure logging is configured
    if not _LOGGING_CONFIGURED:
        configure_logging()

    # Use calling module name if none provided
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "covid_integration")

    # Return logger without adding extra handlers
    return logging.getLogger(name)


def set_log_level(level: str) -> None:
    """
    Change the logging level for all covid_integration loggers.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric_level)


# Convenience functions for different log levels
def log_debug(message: str, logger_name: Optional[str] = None) -> None:
    """Log debug message."""
    get_logger(logger_name).debug(message)


def log_info(message: str, logger_name: Optional[str] = None) -> None:
    """Log info message."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: Optional[str] = None) -> None:
    """Log warning message."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: Optional[str] = None) -> None:
    """Log error message."""
    get_logger(logger_name).error(message)


def log_critical(message: str, logger_name: Optional[str] = None) -> None:
    """Log critical message."""
    get_logger(logger_name).critical(message)


# Don't initialize logging automatically - let modules control when to configure
# configure_logging()
