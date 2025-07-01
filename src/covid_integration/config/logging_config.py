"""
COVID-19 Data Integration - Centralized Logging Configuration

This module provides centralized logging configuration for the entire project.
Ensures consistent log formatting, levels, and output across all modules.
"""

import logging
import sys
from typing import Optional

from .constants import LOG_FORMAT, LOG_LEVEL


def setup_logger(
    name: Optional[str] = None, level: str = LOG_LEVEL, format_string: str = LOG_FORMAT, stream=None
) -> logging.Logger:
    """
    Set up a logger with consistent configuration.

    Args:
        name: Logger name (defaults to calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Log message format
        stream: Output stream (defaults to stdout)

    Returns:
        Configured logger instance
    """
    # Use calling module name if none provided
    if name is None:
        import inspect

        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "covid_integration")

    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger

    # Set level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create handler
    if stream is None:
        stream = sys.stdout

    handler = logging.StreamHandler(stream)
    handler.setLevel(numeric_level)

    # Create formatter
    formatter = logging.Formatter(format_string)
    handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(handler)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with standard configuration.

    Args:
        name: Logger name (defaults to calling module)

    Returns:
        Logger instance
    """
    return setup_logger(name)


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
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        stream=sys.stdout,
        force=True,  # Override any existing configuration
    )

    # Suppress verbose external library logging
    if suppress_external:
        # Common verbose loggers to quiet down
        external_loggers = [
            "urllib3.connectionpool",
            "requests.packages.urllib3",
            "matplotlib",
            "PIL",
            "plotly",
        ]

        for logger_name in external_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def set_log_level(level: str) -> None:
    """
    Change the logging level for all covid_integration loggers.

    Args:
        level: New logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Update all existing loggers that start with covid_integration
    for name in logging.root.manager.loggerDict:
        if isinstance(name, str) and name.startswith("covid_integration"):
            logger = logging.getLogger(name)
            logger.setLevel(numeric_level)
            for handler in logger.handlers:
                handler.setLevel(numeric_level)


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


# Initialize logging when module is imported
configure_logging()
