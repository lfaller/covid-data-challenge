"""
COVID-19 Data Cleaning and Standardization Module

This module handles cleaning and standardizing data from multiple sources to prepare
for integration. Key functions include country name harmonization, data validation,
and handling missing values.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Import centralized constants and logging
try:
    # Relative import (when used as module)
    from .config.constants import (
        COUNTRY_NAME_MAPPING,
        EXCLUDE_REGIONS,
    )
    from .config.logging_config import get_logger
except ImportError:
    # Absolute import (when run directly)
    from covid_integration.config.constants import (
        COUNTRY_NAME_MAPPING,
        EXCLUDE_REGIONS,
    )
    from covid_integration.config.logging_config import get_logger

# Configure logging
logger = get_logger(__name__)


def standardize_country_names(df: pd.DataFrame, source: str = "owid") -> pd.DataFrame:
    """
    Standardize country names for consistent matching across data sources.

    Args:
        df: DataFrame containing country data
        source: Data source ('owid' or 'api') to determine mapping direction

    Returns:
        DataFrame with standardized country names
    """
    df_clean = df.copy()

    if source == "owid":
        # Apply mapping from OWID names to API names for better matching
        df_clean["country_standardized"] = (
            df_clean["country"].map(COUNTRY_NAME_MAPPING).fillna(df_clean["country"])
        )
        logger.info(f"Applied {len(COUNTRY_NAME_MAPPING)} country name mappings for OWID data")
    else:
        # For API data, we should NOT reverse the mapping - keep original names
        # The mapping is designed to convert OWID -> API format, so API data should stay as-is
        df_clean["country_standardized"] = df_clean["country"]
        logger.info("Used original country names for API data (no mapping applied)")

    return df_clean


def filter_valid_countries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out non-country entries (continents, aggregates, etc.).

    Args:
        df: DataFrame with country data

    Returns:
        DataFrame with only valid countries
    """
    initial_count = len(df["country"].unique())

    # Remove excluded regions
    df_filtered = df[~df["country"].isin(EXCLUDE_REGIONS)].copy()

    # Remove any remaining aggregates (often have specific patterns)
    # Remove entries that are clearly not countries
    df_filtered = df_filtered[
        ~df_filtered["country"].str.contains(
            r"income|OECD|Union|International", case=False, na=False
        )
    ]

    final_count = len(df_filtered["country"].unique())
    logger.info(
        f"Filtered countries: {initial_count} -> {final_count} "
        f"(removed {initial_count - final_count} non-country entries)"
    )

    return df_filtered


def validate_data_quality(df: pd.DataFrame, source: str) -> Dict:
    """
    Assess data quality and identify potential issues.

    Args:
        df: DataFrame to validate
        source: Data source name for logging

    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}

    # Basic stats
    quality_metrics["total_rows"] = len(df)
    quality_metrics["total_countries"] = df["country"].nunique()
    quality_metrics["date_range"] = None

    if "date" in df.columns:
        quality_metrics["date_range"] = (df["date"].min(), df["date"].max())
        quality_metrics["date_span_days"] = (df["date"].max() - df["date"].min()).days

    # Missing data analysis
    missing_data = {}
    numeric_columns = df.select_dtypes(include=[np.number]).columns

    for col in numeric_columns:
        missing_count = df[col].isnull().sum()
        missing_pct = (missing_count / len(df)) * 100
        missing_data[col] = {
            "missing_count": missing_count,
            "missing_percentage": round(missing_pct, 2),
        }

    quality_metrics["missing_data"] = missing_data

    # Data consistency checks
    if "population" in df.columns:
        # Check for reasonable population values
        pop_outliers = df[(df["population"] < 1000) | (df["population"] > 2e9)]
        quality_metrics["population_outliers"] = len(pop_outliers)

    # Check for negative values in case/death columns
    case_death_cols = [
        col
        for col in df.columns
        if any(x in col.lower() for x in ["case", "death", "test"])
        and df[col].dtype in ["int64", "float64"]
    ]

    negative_values = {}
    for col in case_death_cols:
        negative_count = (df[col] < 0).sum()
        if negative_count > 0:
            negative_values[col] = negative_count

    quality_metrics["negative_values"] = negative_values

    logger.info(
        f"Quality assessment for {source}: {quality_metrics['total_countries']} countries, "
        f"{quality_metrics['total_rows']} rows"
    )

    return quality_metrics


def clean_owid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare OWID historical data.

    Args:
        df: Raw OWID DataFrame

    Returns:
        Cleaned OWID DataFrame
    """
    logger.info("Cleaning OWID historical data...")

    # Standardize country names
    df_clean = standardize_country_names(df, source="owid")

    # Filter to valid countries only
    df_clean = filter_valid_countries(df_clean)

    # Handle missing values in key columns
    # For cumulative columns, forward fill within each country
    cumulative_cols = [
        "total_cases",
        "total_deaths",
        "total_tests",
        "people_vaccinated",
        "people_fully_vaccinated",
    ]

    for col in cumulative_cols:
        if col in df_clean.columns:
            # Forward fill within each country group
            df_clean[col] = df_clean.groupby("country")[col].ffill()

    # Calculate daily changes where missing
    if "new_cases" not in df_clean.columns and "total_cases" in df_clean.columns:
        df_clean["new_cases"] = df_clean.groupby("country")["total_cases"].diff()
        df_clean["new_cases"] = df_clean["new_cases"].fillna(0).clip(lower=0)

    if "new_deaths" not in df_clean.columns and "total_deaths" in df_clean.columns:
        df_clean["new_deaths"] = df_clean.groupby("country")["total_deaths"].diff()
        df_clean["new_deaths"] = df_clean["new_deaths"].fillna(0).clip(lower=0)

    # Add data source identifier
    df_clean["data_source"] = "owid_historical"

    logger.info(
        f"OWID data cleaned: {len(df_clean)} rows, {df_clean['country'].nunique()} countries"
    )

    return df_clean


def clean_api_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare disease.sh API current data.

    Args:
        df: Raw API DataFrame

    Returns:
        Cleaned API DataFrame
    """
    logger.info("Cleaning disease.sh API current data...")

    # Standardize country names
    df_clean = standardize_country_names(df, source="api")

    # Filter to valid countries only
    df_clean = filter_valid_countries(df_clean)

    # Handle any negative values (shouldn't be any, but safety check)
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "latitude" and col != "longitude":  # Geographic coordinates can be negative
            df_clean[col] = df_clean[col].clip(lower=0)

    # Calculate rates and ratios
    if "current_cases" in df_clean.columns and "population" in df_clean.columns:
        df_clean["cases_per_100k"] = (
            df_clean["current_cases"] / df_clean["population"] * 100000
        ).round(2)

    if "current_deaths" in df_clean.columns and "population" in df_clean.columns:
        df_clean["deaths_per_100k"] = (
            df_clean["current_deaths"] / df_clean["population"] * 100000
        ).round(2)

    if "current_deaths" in df_clean.columns and "current_cases" in df_clean.columns:
        # Avoid division by zero
        df_clean["case_fatality_rate"] = np.where(
            df_clean["current_cases"] > 0,
            (df_clean["current_deaths"] / df_clean["current_cases"] * 100).round(3),
            0,
        )

    # Add data source identifier
    df_clean["data_source"] = "disease_sh_current"

    logger.info(
        f"API data cleaned: {len(df_clean)} rows, {df_clean['country'].nunique()} countries"
    )

    return df_clean


def identify_matching_countries(owid_df: pd.DataFrame, api_df: pd.DataFrame) -> Dict:
    """
    Identify which countries can be matched between the two datasets.

    Args:
        owid_df: Cleaned OWID DataFrame
        api_df: Cleaned API DataFrame

    Returns:
        Dictionary with matching statistics and country lists
    """
    # Use standardized country names for matching
    owid_countries = set(owid_df["country_standardized"].unique())
    api_countries = set(api_df["country_standardized"].unique())

    # Find matches and mismatches
    matched_countries = owid_countries.intersection(api_countries)
    owid_only = owid_countries - api_countries
    api_only = api_countries - owid_countries

    matching_stats = {
        "total_matched": len(matched_countries),
        "owid_only_count": len(owid_only),
        "api_only_count": len(api_only),
        "match_rate_owid": len(matched_countries) / len(owid_countries) * 100,
        "match_rate_api": len(matched_countries) / len(api_countries) * 100,
        "matched_countries": sorted(list(matched_countries)),
        "owid_only": sorted(list(owid_only)),
        "api_only": sorted(list(api_only)),
    }

    logger.info(
        f"Country matching: {len(matched_countries)} matched, "
        f"{len(owid_only)} OWID-only, {len(api_only)} API-only"
    )
    logger.info(
        f"Match rates: OWID {matching_stats['match_rate_owid']:.1f}%, "
        f"API {matching_stats['match_rate_api']:.1f}%"
    )

    return matching_stats


def clean_all_data(
    owid_df: pd.DataFrame, api_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Clean both datasets and assess matching potential.

    Args:
        owid_df: Raw OWID DataFrame
        api_df: Raw API DataFrame

    Returns:
        Tuple of (cleaned_owid_df, cleaned_api_df, quality_report)
    """
    logger.info("Starting comprehensive data cleaning process...")

    # Clean individual datasets
    owid_clean = clean_owid_data(owid_df)
    api_clean = clean_api_data(api_df)

    # Assess data quality
    owid_quality = validate_data_quality(owid_clean, "OWID")
    api_quality = validate_data_quality(api_clean, "disease.sh API")

    # Identify matching countries
    matching_stats = identify_matching_countries(owid_clean, api_clean)

    # Compile comprehensive quality report
    quality_report = {
        "owid_quality": owid_quality,
        "api_quality": api_quality,
        "country_matching": matching_stats,
        "cleaning_summary": {
            "owid_countries_before": owid_df["country"].nunique(),
            "owid_countries_after": owid_clean["country"].nunique(),
            "api_countries_before": api_df["country"].nunique(),
            "api_countries_after": api_clean["country"].nunique(),
            "country_mappings_applied": len(COUNTRY_NAME_MAPPING),
        },
    }

    logger.info("Data cleaning completed successfully!")

    return owid_clean, api_clean, quality_report


if __name__ == "__main__":
    # Test the cleaning functions with sample data
    from data_loader import load_all_data

    try:
        # Load raw data
        owid_raw, api_raw, _ = load_all_data()

        # Clean the data
        owid_clean, api_clean, quality_report = clean_all_data(owid_raw, api_raw)

        print("\n=== Data Cleaning Results ===")
        print(
            f"OWID: {owid_raw['country'].nunique()} -> {owid_clean['country'].nunique()} countries"
        )
        print(f"API: {api_raw['country'].nunique()} -> {api_clean['country'].nunique()} countries")
        print(f"Matched countries: {quality_report['country_matching']['total_matched']}")

        print("\n=== Sample Cleaned Data ===")
        print("OWID cleaned sample:")
        print(
            owid_clean[
                ["country", "country_standardized", "date", "total_cases", "data_source"]
            ].head()
        )

        print("\nAPI cleaned sample:")
        print(
            api_clean[
                [
                    "country",
                    "country_standardized",
                    "current_cases",
                    "cases_per_100k",
                    "data_source",
                ]
            ].head()
        )

        print("\n=== Matching Summary ===")
        matching = quality_report["country_matching"]
        print(f"Total matched: {matching['total_matched']}")
        print(f"OWID-only: {matching['owid_only_count']} (e.g., {matching['owid_only'][:5]})")
        print(f"API-only: {matching['api_only_count']} (e.g., {matching['api_only'][:5]})")

    except Exception as e:
        logger.error(f"Data cleaning test failed: {e}")
        raise
