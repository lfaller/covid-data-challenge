"""
COVID-19 Data Integration - Data Filtering Utilities

This module provides centralized filtering functions for removing non-country
aggregates and regions from COVID-19 datasets. Eliminates code duplication
across data_loader and data_cleaner modules.
"""

from typing import List, Optional, Set

import pandas as pd

from covid_integration import get_constants, get_logger

# Get constants
EXCLUDE_REGIONS = get_constants('EXCLUDE_REGIONS')

# Configure logging
logger = get_logger(__name__)


def filter_non_countries(
    df: pd.DataFrame,
    country_col: str = "country",
    iso_code_col: str = "iso_code",
    exclude_regions: Optional[Set[str]] = None,
    exclude_patterns: Optional[List[str]] = None,
    exclude_iso_codes: Optional[List[str]] = None,
    log_results: bool = True
) -> pd.DataFrame:
    """
    Remove non-country aggregates and regions from dataset.

    This function provides a centralized way to filter out world aggregates,
    continental summaries, income group aggregates, and other non-country
    entries from COVID-19 datasets.

    Args:
        df: DataFrame to filter
        country_col: Name of the country column (default: "country")
        iso_code_col: Name of the ISO code column (default: "iso_code")
        exclude_regions: Set of country names to exclude (default: uses EXCLUDE_REGIONS constant)
        exclude_patterns: List of regex patterns to exclude (default: income/OECD/Union/International)
        exclude_iso_codes: List of ISO codes to exclude (default: OWID aggregates)
        log_results: Whether to log filtering results (default: True)

    Returns:
        DataFrame with non-country entries removed

    Example:
        >>> df_filtered = filter_non_countries(df)
        >>> df_filtered = filter_non_countries(df, country_col="location")
        >>> df_filtered = filter_non_countries(
        ...     df,
        ...     exclude_patterns=[r"income", r"World"],
        ...     log_results=False
        ... )
    """
    if exclude_regions is None:
        exclude_regions = EXCLUDE_REGIONS

    if exclude_patterns is None:
        exclude_patterns = [r"income|OECD|Union|International"]

    if exclude_iso_codes is None:
        # OWID-specific aggregate codes
        exclude_iso_codes = ["OWID_WRL", "OWID_HIC", "OWID_LIC", "OWID_LMC", "OWID_UMC"]

    initial_count = len(df[country_col].unique()) if country_col in df.columns else 0
    df_filtered = df.copy()

    # Filter by country names
    if country_col in df.columns and exclude_regions:
        df_filtered = df_filtered[~df_filtered[country_col].isin(exclude_regions)]

        # Apply pattern-based exclusions
        for pattern in exclude_patterns:
            df_filtered = df_filtered[
                ~df_filtered[country_col].str.contains(pattern, case=False, na=False)
            ]

    # Filter by ISO codes
    if iso_code_col in df.columns and exclude_iso_codes:
        df_filtered = df_filtered[~df_filtered[iso_code_col].isin(exclude_iso_codes)]

    final_count = len(df_filtered[country_col].unique()) if country_col in df_filtered.columns else 0

    if log_results and initial_count > 0:
        logger.info(
            f"Filtered countries: {initial_count} -> {final_count} "
            f"(removed {initial_count - final_count} non-country entries)"
        )

    return df_filtered


def filter_valid_countries(df: pd.DataFrame, country_col: str = "country") -> pd.DataFrame:
    """
    Filter out non-country entries (continents, aggregates, etc.).

    This is a convenience wrapper around filter_non_countries() that uses
    default parameters for backward compatibility with existing code.

    Args:
        df: DataFrame with country data
        country_col: Name of the country column (default: "country")

    Returns:
        DataFrame with only valid countries

    Example:
        >>> df_clean = filter_valid_countries(df)
    """
    return filter_non_countries(df, country_col=country_col, log_results=True)
