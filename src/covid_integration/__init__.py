"""
COVID-19 Data Integration Package

This package provides utilities for loading, cleaning, merging, and visualizing
COVID-19 data from multiple sources.
"""

# Import utilities for handling relative/absolute imports
from typing import Any, Dict


def get_config_imports() -> Dict[str, Any]:
    """
    Centralized import handler for config modules.

    This function handles both relative imports (when used as a module)
    and absolute imports (when run directly) to avoid code duplication
    across all modules in the package.

    Returns:
        Dictionary containing all imported config objects
    """
    imports = {}

    try:
        # Relative import (when used as module)
        from .config import constants, logging_config
        imports['constants'] = constants
        imports['logging_config'] = logging_config
    except ImportError:
        # Absolute import (when run directly)
        from covid_integration.config import constants, logging_config
        imports['constants'] = constants
        imports['logging_config'] = logging_config

    return imports


def get_logger(name: str):
    """
    Get a configured logger instance.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    imports = get_config_imports()
    return imports['logging_config'].get_logger(name)


# Convenience function to get specific constants
def get_constants(*attr_names):
    """
    Get specific constants from the config module.

    Args:
        *attr_names: Names of constants to retrieve

    Returns:
        Single constant if one name provided, tuple of constants if multiple

    Example:
        url = get_constants('OWID_CSV_URL')
        url1, url2 = get_constants('OWID_CSV_URL', 'DISEASE_SH_API_URL')
    """
    imports = get_config_imports()
    constants_module = imports['constants']

    if len(attr_names) == 1:
        return getattr(constants_module, attr_names[0])

    return tuple(getattr(constants_module, attr) for attr in attr_names)
