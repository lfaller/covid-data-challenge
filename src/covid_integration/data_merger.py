"""
COVID-19 Data Integration and Merging Module

This module handles the integration of historical OWID data with current disease.sh API data
to create unified datasets for analysis. Key functions include temporal alignment,
data harmonization, and intelligent gap filling.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Import centralized constants
try:
    # Relative import (when used as module)
    from .config.constants import DEFAULT_TREND_WINDOW_DAYS
except ImportError:
    # Absolute import (when run directly)
    from covid_integration.config.constants import DEFAULT_TREND_WINDOW_DAYS

# Configure logging
logger = logging.getLogger(__name__)


def create_country_mapping_table(owid_df: pd.DataFrame, api_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a mapping table showing which countries exist in each dataset.

    Args:
        owid_df: Cleaned OWID DataFrame
        api_df: Cleaned API DataFrame

    Returns:
        DataFrame with country mapping information
    """
    # Get unique countries from each source
    owid_countries = set(owid_df["country_standardized"].unique())
    api_countries = set(api_df["country_standardized"].unique())

    # Create comprehensive country list
    all_countries = owid_countries.union(api_countries)

    mapping_data = []
    for country in sorted(all_countries):
        mapping_data.append(
            {
                "country": country,
                "in_owid": country in owid_countries,
                "in_api": country in api_countries,
                "can_merge": country in owid_countries and country in api_countries,
            }
        )

    mapping_df = pd.DataFrame(mapping_data)
    logger.info(
        f"Created country mapping table: {len(mapping_df)} total countries, "
        f"{mapping_df['can_merge'].sum()} can be merged"
    )

    return mapping_df


def align_temporal_data(
    owid_df: pd.DataFrame, api_df: pd.DataFrame, alignment_strategy: str = "latest"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align temporal aspects of the datasets for meaningful comparison.

    Args:
        owid_df: Historical OWID data
        api_df: Current API data
        alignment_strategy: How to align dates ('latest', 'specific_date')

    Returns:
        Tuple of temporally aligned dataframes
    """
    logger.info(f"Aligning temporal data using strategy: {alignment_strategy}")

    if alignment_strategy == "latest":
        # Get the latest date from OWID data for each country
        latest_owid = (
            owid_df.groupby("country_standardized")
            .agg(
                {
                    "date": "max",
                    "total_cases": "last",
                    "total_deaths": "last",
                    "population": "last",
                    "iso_code": "last",
                    "country": "last",
                }
            )
            .reset_index()
        )

        # Add suffix to distinguish from API data
        latest_owid.columns = [
            col if col in ["country_standardized", "country", "iso_code"] else f"owid_{col}"
            for col in latest_owid.columns
        ]

        # API data represents current state, add prefix for clarity
        api_aligned = api_df.copy()
        api_aligned.columns = [
            col if col in ["country_standardized", "country", "iso_code"] else f"api_{col}"
            for col in api_aligned.columns
        ]

        logger.info(
            f"Temporal alignment complete: OWID latest data from {latest_owid['owid_date'].max()}"
        )
        return latest_owid, api_aligned

    else:
        raise ValueError(f"Unsupported alignment strategy: {alignment_strategy}")


def calculate_trend_metrics(
    owid_df: pd.DataFrame, window_days: int = DEFAULT_TREND_WINDOW_DAYS
) -> pd.DataFrame:
    """
    Calculate trend metrics from historical data for the last N days.

    Args:
        owid_df: Historical OWID data
        window_days: Number of recent days to analyze for trends

    Returns:
        DataFrame with trend metrics by country
    """
    logger.info(f"Calculating trend metrics using {window_days}-day window")

    # Get the most recent date in the dataset
    max_date = owid_df["date"].max()
    cutoff_date = max_date - timedelta(days=window_days)

    # Filter to recent data
    recent_data = owid_df[owid_df["date"] >= cutoff_date].copy()

    trend_metrics = []

    for country in recent_data["country_standardized"].unique():
        country_data = recent_data[recent_data["country_standardized"] == country].sort_values(
            "date"
        )

        if len(country_data) < 2:
            continue

        # Calculate trends
        first_cases = (
            country_data["total_cases"].iloc[0]
            if pd.notna(country_data["total_cases"].iloc[0])
            else 0
        )
        last_cases = (
            country_data["total_cases"].iloc[-1]
            if pd.notna(country_data["total_cases"].iloc[-1])
            else 0
        )

        first_deaths = (
            country_data["total_deaths"].iloc[0]
            if pd.notna(country_data["total_deaths"].iloc[0])
            else 0
        )
        last_deaths = (
            country_data["total_deaths"].iloc[-1]
            if pd.notna(country_data["total_deaths"].iloc[-1])
            else 0
        )

        # Calculate daily averages for new cases/deaths
        if "new_cases" in country_data.columns:
            avg_daily_cases = (
                country_data["new_cases"].mean()
                if not country_data["new_cases"].isna().all()
                else 0
            )
        else:
            avg_daily_cases = 0

        if "new_deaths" in country_data.columns:
            avg_daily_deaths = (
                country_data["new_deaths"].mean()
                if not country_data["new_deaths"].isna().all()
                else 0
            )
        else:
            avg_daily_deaths = 0

        # Calculate percentage change
        cases_pct_change = (
            ((last_cases - first_cases) / first_cases * 100) if first_cases > 0 else 0
        )
        deaths_pct_change = (
            ((last_deaths - first_deaths) / first_deaths * 100) if first_deaths > 0 else 0
        )

        trend_metrics.append(
            {
                "country_standardized": country,
                "trend_window_days": window_days,
                "trend_start_date": country_data["date"].min(),
                "trend_end_date": country_data["date"].max(),
                "cases_change_absolute": last_cases - first_cases,
                "deaths_change_absolute": last_deaths - first_deaths,
                "cases_change_percent": round(cases_pct_change, 2),
                "deaths_change_percent": round(deaths_pct_change, 2),
                "avg_daily_new_cases": round(avg_daily_cases, 1),
                "avg_daily_new_deaths": round(avg_daily_deaths, 1),
                "data_points_available": len(country_data),
            }
        )

    trend_df = pd.DataFrame(trend_metrics)
    logger.info(f"Calculated trends for {len(trend_df)} countries")

    return trend_df


def merge_datasets(
    owid_df: pd.DataFrame, api_df: pd.DataFrame, include_trends: bool = True
) -> pd.DataFrame:
    """
    Merge OWID historical data with API current data.

    Args:
        owid_df: Cleaned OWID DataFrame
        api_df: Cleaned API DataFrame
        include_trends: Whether to include trend analysis

    Returns:
        Merged DataFrame with combined metrics
    """
    logger.info("Starting dataset merge process...")

    # Step 1: Create country mapping
    country_mapping = create_country_mapping_table(owid_df, api_df)
    mergeable_countries = country_mapping[country_mapping["can_merge"]]["country"].tolist()

    logger.info(f"Found {len(mergeable_countries)} countries that can be merged")

    # Step 2: Temporal alignment
    owid_aligned, api_aligned = align_temporal_data(owid_df, api_df)

    # Step 3: Calculate trends if requested
    if include_trends:
        trend_metrics = calculate_trend_metrics(owid_df)

    # Step 4: Perform the merge
    merged_df = pd.merge(
        owid_aligned,
        api_aligned,
        on="country_standardized",
        how="inner",
        suffixes=("_owid_meta", "_api_meta"),
    )

    # Step 5: Add trend metrics if available
    if include_trends:
        merged_df = pd.merge(merged_df, trend_metrics, on="country_standardized", how="left")

    # Step 6: Calculate derived metrics
    merged_df = calculate_derived_metrics(merged_df)

    # Step 7: Add metadata
    merged_df["merge_timestamp"] = datetime.now()

    # Debug: Check what date columns we actually have
    date_columns = [col for col in merged_df.columns if "date" in col.lower()]
    logger.info(f"Available date columns: {date_columns}")
    logger.info(
        f"Column dtypes for date columns: {merged_df[date_columns].dtypes.to_dict() if date_columns else 'None'}"
    )

    # Fix datetime conversion issue - be more defensive
    owid_date_col = None
    for col in ["owid_date", "date", "owid_latest_date"]:
        if col in merged_df.columns:
            owid_date_col = col
            break

    if owid_date_col:
        # Ensure it's datetime
        merged_df[owid_date_col] = pd.to_datetime(merged_df[owid_date_col])
        merged_df["owid_data_age_days"] = (datetime.now() - merged_df[owid_date_col]).dt.days
        logger.info(f"Used {owid_date_col} for age calculation")
    else:
        merged_df["owid_data_age_days"] = None
        logger.warning("No OWID date column found for age calculation")

    logger.info(f"Merge completed: {len(merged_df)} countries in final dataset")

    return merged_df


def calculate_derived_metrics(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate additional metrics from the merged dataset.

    Args:
        merged_df: Merged DataFrame

    Returns:
        DataFrame with additional calculated metrics
    """
    df = merged_df.copy()

    # Data freshness comparison
    if "owid_total_cases" in df.columns and "api_current_cases" in df.columns:
        df["cases_data_gap"] = df["api_current_cases"] - df["owid_total_cases"]
        df["cases_data_gap_percent"] = np.where(
            df["owid_total_cases"] > 0,
            (df["cases_data_gap"] / df["owid_total_cases"] * 100).round(2),
            0,
        )

    if "owid_total_deaths" in df.columns and "api_current_deaths" in df.columns:
        df["deaths_data_gap"] = df["api_current_deaths"] - df["owid_total_deaths"]
        df["deaths_data_gap_percent"] = np.where(
            df["owid_total_deaths"] > 0,
            (df["deaths_data_gap"] / df["owid_total_deaths"] * 100).round(2),
            0,
        )

    # Population-adjusted metrics comparison
    if "owid_population" in df.columns:
        if "api_current_cases" in df.columns:
            df["current_cases_per_100k"] = (
                df["api_current_cases"] / df["owid_population"] * 100000
            ).round(2)
        if "api_current_deaths" in df.columns:
            df["current_deaths_per_100k"] = (
                df["api_current_deaths"] / df["owid_population"] * 100000
            ).round(2)

    # Case fatality rate from current data
    if "api_current_cases" in df.columns and "api_current_deaths" in df.columns:
        df["current_case_fatality_rate"] = np.where(
            df["api_current_cases"] > 0,
            (df["api_current_deaths"] / df["api_current_cases"] * 100).round(3),
            0,
        )

    logger.info("Calculated derived metrics for merged dataset")

    return df


def generate_integration_summary(
    merged_df: pd.DataFrame, country_mapping: pd.DataFrame = None
) -> Dict:
    """
    Generate a comprehensive summary of the integration process.

    Args:
        merged_df: Final merged DataFrame
        country_mapping: Country mapping table (optional)

    Returns:
        Dictionary with integration summary statistics
    """
    summary = {
        "integration_timestamp": datetime.now().isoformat(),
        "total_countries_merged": len(merged_df),
        "data_completeness": {},
        "data_quality_flags": {},
        "temporal_analysis": {},
        "top_countries_by_cases": [],
        "countries_with_trends": 0,
    }

    # Data completeness analysis
    key_columns = [
        "owid_total_cases",
        "api_current_cases",
        "owid_total_deaths",
        "api_current_deaths",
    ]
    for col in key_columns:
        if col in merged_df.columns:
            non_null_count = merged_df[col].notna().sum()
            summary["data_completeness"][col] = {
                "available": int(non_null_count),
                "missing": int(len(merged_df) - non_null_count),
                "completeness_percent": round(non_null_count / len(merged_df) * 100, 1),
            }

    # Data quality flags
    if "cases_data_gap_percent" in merged_df.columns:
        large_gaps = (merged_df["cases_data_gap_percent"].abs() > 10).sum()
        summary["data_quality_flags"]["countries_with_large_case_gaps"] = int(large_gaps)

    if "owid_data_age_days" in merged_df.columns:
        old_data = (merged_df["owid_data_age_days"] > 90).sum()
        summary["data_quality_flags"]["countries_with_old_owid_data"] = int(old_data)
        summary["temporal_analysis"]["avg_owid_data_age_days"] = float(
            merged_df["owid_data_age_days"].mean()
        )

    # Top countries by current cases
    if "api_current_cases" in merged_df.columns:
        top_countries = merged_df.nlargest(10, "api_current_cases")[
            ["country_standardized", "api_current_cases", "current_cases_per_100k"]
        ].to_dict("records")
        summary["top_countries_by_cases"] = top_countries

    # Trend analysis availability
    if "avg_daily_new_cases" in merged_df.columns:
        summary["countries_with_trends"] = merged_df["avg_daily_new_cases"].notna().sum()

    logger.info(f"Generated integration summary for {summary['total_countries_merged']} countries")

    return summary


def integrate_covid_data(
    owid_df: pd.DataFrame, api_df: pd.DataFrame, output_dir: str = "outputs"
) -> Tuple[pd.DataFrame, Dict]:
    """
    Main integration function that combines all steps.

    Args:
        owid_df: Cleaned OWID DataFrame
        api_df: Cleaned API DataFrame
        output_dir: Directory to save outputs

    Returns:
        Tuple of (merged_dataframe, integration_summary)
    """
    logger.info("Starting complete COVID data integration process...")

    # Perform the merge
    merged_df = merge_datasets(owid_df, api_df, include_trends=True)

    # Generate country mapping table
    country_mapping = create_country_mapping_table(owid_df, api_df)

    # Generate summary
    integration_summary = generate_integration_summary(merged_df, country_mapping)

    # Log key results
    logger.info(f"Integration complete: {len(merged_df)} countries successfully merged")
    logger.info(f"Data completeness: {integration_summary['data_completeness']}")

    return merged_df, integration_summary


if __name__ == "__main__":
    # Test the integration functions
    from data_cleaner import clean_all_data
    from data_loader import load_all_data

    try:
        # Load and clean data
        logger.info("Loading raw data...")
        owid_raw, api_raw, _ = load_all_data()

        logger.info("Cleaning data...")
        owid_clean, api_clean, _ = clean_all_data(owid_raw, api_raw)

        # Perform integration
        logger.info("Starting integration...")
        merged_data, summary = integrate_covid_data(owid_clean, api_clean)

        print("\n=== Integration Results ===")
        print(f"Countries merged: {len(merged_data)}")
        print(f"Columns in merged dataset: {len(merged_data.columns)}")

        print("\n=== Sample Merged Data ===")
        display_columns = [
            "country_standardized",
            "owid_total_cases",
            "api_current_cases",
            "cases_data_gap",
            "current_cases_per_100k",
            "avg_daily_new_cases",
        ]
        available_columns = [col for col in display_columns if col in merged_data.columns]
        print(merged_data[available_columns].head(10))

        print("\n=== Integration Summary ===")
        print(f"Total countries merged: {summary['total_countries_merged']}")
        print(f"Countries with trend data: {summary['countries_with_trends']}")
        print(
            f"Average OWID data age: {summary['temporal_analysis'].get('avg_owid_data_age_days', 'N/A')} days"
        )

        if summary["top_countries_by_cases"]:
            print("\nTop 5 countries by current cases:")
            for i, country in enumerate(summary["top_countries_by_cases"][:5], 1):
                print(
                    f"{i}. {country['country_standardized']}: {country['api_current_cases']:,} cases"
                )

    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        raise
