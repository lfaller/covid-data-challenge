"""
Test suite for COVID-19 data integration modules.

This test suite covers:
- Data loading functionality
- Data cleaning and standardization
- Country name mapping
- Data quality validation
"""

from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import requests

from covid_integration.data_cleaner import (
    COUNTRY_NAME_MAPPING,
    EXCLUDE_REGIONS,
    clean_api_data,
    clean_owid_data,
    filter_valid_countries,
    identify_matching_countries,
    standardize_country_names,
    validate_data_quality,
)

# Import modules to test
from covid_integration.data_loader import (
    compare_data_sources,
    load_disease_sh_data,
    parse_disease_sh_json,
)
from covid_integration.data_merger import (
    align_temporal_data,
    calculate_derived_metrics,
    calculate_trend_metrics,
    create_country_mapping_table,
    generate_integration_summary,
    integrate_covid_data,
    merge_datasets,
)


class TestDataLoader:
    """Test cases for data loading functionality."""

    def test_parse_disease_sh_json(self):
        """Test parsing of disease.sh API JSON response."""
        # Mock API response data
        mock_api_data = [
            {
                "country": "Afghanistan",
                "countryInfo": {
                    "iso3": "AFG",
                    "iso2": "AF",
                    "_id": 4,
                    "lat": 33,
                    "long": 65,
                    "flag": "flag_url",
                },
                "population": 40000000,
                "cases": 234174,
                "deaths": 7896,
                "recovered": 180000,
                "active": 46278,
                "critical": 31,
                "casesPerOneMillion": 5854,
                "deathsPerOneMillion": 197,
                "tests": 1000000,
                "testsPerOneMillion": 25000,
                "todayCases": 0,
                "todayDeaths": 0,
                "todayRecovered": 0,
                "updated": 1640995200000,  # Timestamp
            }
        ]

        df = parse_disease_sh_json(mock_api_data)

        # Assertions
        assert len(df) == 1
        assert df.iloc[0]["country"] == "Afghanistan"
        assert df.iloc[0]["iso_code"] == "AFG"
        assert df.iloc[0]["population"] == 40000000
        assert df.iloc[0]["current_cases"] == 234174
        assert "last_updated" in df.columns
        assert pd.notna(df.iloc[0]["last_updated"])

    def test_compare_data_sources(self):
        """Test comparison functionality between data sources."""
        # Create mock dataframes
        owid_df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Albania", "Algeria", "Test Country"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]),
            }
        )

        api_df = pd.DataFrame(
            {"country": ["Afghanistan", "Albania", "Different Country", "Another Country"]}
        )

        stats = compare_data_sources(owid_df, api_df)

        # Assertions
        assert stats["owid_total_countries"] == 4
        assert stats["api_total_countries"] == 4
        assert stats["common_countries"] == 2  # Afghanistan, Albania
        assert stats["owid_only_count"] == 2  # Algeria, Test Country
        assert stats["api_only_count"] == 2  # Different Country, Another Country
        assert "latest_owid_date" in stats

    @patch("requests.get")
    def test_load_disease_sh_data_success(self, mock_get):
        """Test successful API data loading."""
        # Mock successful API response
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = [
            {
                "country": "Test Country",
                "countryInfo": {"iso3": "TST", "iso2": "TS", "_id": 999},
                "population": 1000000,
                "cases": 1000,
                "deaths": 10,
                "updated": 1640995200000,
            }
        ]
        mock_get.return_value = mock_response

        df = load_disease_sh_data("http://test-url.com")

        # Assertions
        assert len(df) == 1
        assert df.iloc[0]["country"] == "Test Country"
        mock_get.assert_called_once()

    @patch("requests.get")
    def test_load_disease_sh_data_api_error(self, mock_get):
        """Test API error handling."""
        # Mock API error
        mock_get.side_effect = requests.exceptions.RequestException("API Error")

        with pytest.raises(requests.exceptions.RequestException):
            load_disease_sh_data("http://test-url.com")


class TestDataCleaner:
    """Test cases for data cleaning and standardization."""

    def test_standardize_country_names_owid(self):
        """Test country name standardization for OWID data."""
        df = pd.DataFrame(
            {"country": ["Bosnia and Herzegovina", "Cape Verde", "United States", "Afghanistan"]}
        )

        df_standardized = standardize_country_names(df, source="owid")

        # Check that mappings are applied
        assert df_standardized.loc[0, "country_standardized"] == "Bosnia"
        assert df_standardized.loc[1, "country_standardized"] == "Cabo Verde"
        assert df_standardized.loc[2, "country_standardized"] == "USA"
        assert df_standardized.loc[3, "country_standardized"] == "Afghanistan"  # No mapping

    def test_standardize_country_names_api(self):
        """Test country name standardization for API data."""
        df = pd.DataFrame({"country": ["Bosnia", "Cabo Verde", "USA", "Afghanistan"]})

        df_standardized = standardize_country_names(df, source="api")

        # Check reverse mapping (though in this case, should mostly stay the same)
        assert "country_standardized" in df_standardized.columns
        assert len(df_standardized) == 4

    def test_filter_valid_countries(self):
        """Test filtering of non-country entries."""
        df = pd.DataFrame(
            {
                "country": [
                    "Afghanistan",
                    "World",
                    "Europe",
                    "Albania",
                    "High income",
                    "Diamond Princess",
                ]
            }
        )

        df_filtered = filter_valid_countries(df)

        # Should remove aggregates and non-countries
        remaining_countries = df_filtered["country"].tolist()
        assert "Afghanistan" in remaining_countries
        assert "Albania" in remaining_countries
        assert "World" not in remaining_countries
        assert "Europe" not in remaining_countries
        assert "High income" not in remaining_countries
        assert "Diamond Princess" not in remaining_countries

    def test_validate_data_quality(self):
        """Test data quality validation."""
        df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Albania"] * 5,
                "date": pd.date_range("2024-01-01", periods=10),
                "population": [40000000, 3000000] * 5,
                "total_cases": [1000, 500, np.nan, 600, 700, 800, np.nan, 900, 1000, 1100],
                "total_deaths": [10, 5, 15, np.nan, 25, 30, 35, np.nan, 45, 50],
            }
        )

        quality_metrics = validate_data_quality(df, "test_source")

        # Assertions
        assert quality_metrics["total_rows"] == 10
        assert quality_metrics["total_countries"] == 2
        assert "date_range" in quality_metrics
        assert "missing_data" in quality_metrics
        assert "total_cases" in quality_metrics["missing_data"]
        assert quality_metrics["missing_data"]["total_cases"]["missing_count"] == 2
        assert quality_metrics["missing_data"]["total_cases"]["missing_percentage"] == 20.0

    def test_clean_owid_data(self):
        """Test OWID data cleaning pipeline."""
        df = pd.DataFrame(
            {
                "country": ["Afghanistan", "World", "Albania", "Europe"],
                "date": pd.date_range("2024-01-01", periods=4),
                "iso_code": ["AFG", "OWID_WRL", "ALB", "OWID_EUR"],
                "population": [40000000, 7800000000, 3000000, 750000000],
                "total_cases": [1000, 50000000, 500, 25000000],
                "total_deaths": [50, 1000000, 25, 500000],
            }
        )

        df_cleaned = clean_owid_data(df)

        # Should remove aggregates and add standardization
        assert "World" not in df_cleaned["country"].values
        assert "Europe" not in df_cleaned["country"].values
        assert "Afghanistan" in df_cleaned["country"].values
        assert "Albania" in df_cleaned["country"].values
        assert "country_standardized" in df_cleaned.columns
        assert "data_source" in df_cleaned.columns
        assert df_cleaned["data_source"].iloc[0] == "owid_historical"

    def test_clean_api_data(self):
        """Test API data cleaning pipeline."""
        df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Albania", "Diamond Princess"],
                "iso_code": ["AFG", "ALB", "XXX"],
                "population": [40000000, 3000000, 3711],
                "current_cases": [1000, 500, 712],
                "current_deaths": [50, 25, 14],
                "latitude": [33.0, 41.0, 35.0],
                "longitude": [65.0, 20.0, 139.0],
            }
        )

        df_cleaned = clean_api_data(df)

        # Should remove non-countries and add calculations
        assert "Diamond Princess" not in df_cleaned["country"].values
        assert "Afghanistan" in df_cleaned["country"].values
        assert "country_standardized" in df_cleaned.columns
        assert "cases_per_100k" in df_cleaned.columns
        assert "deaths_per_100k" in df_cleaned.columns
        assert "case_fatality_rate" in df_cleaned.columns
        assert "data_source" in df_cleaned.columns
        assert df_cleaned["data_source"].iloc[0] == "disease_sh_current"

        # Check calculated metrics
        afghanistan_row = df_cleaned[df_cleaned["country"] == "Afghanistan"].iloc[0]
        expected_cases_per_100k = (1000 / 40000000) * 100000
        assert abs(afghanistan_row["cases_per_100k"] - expected_cases_per_100k) < 0.1

    def test_identify_matching_countries(self):
        """Test country matching between datasets."""
        owid_df = pd.DataFrame(
            {"country_standardized": ["Afghanistan", "Albania", "Algeria", "Unique_OWID"]}
        )

        api_df = pd.DataFrame(
            {"country_standardized": ["Afghanistan", "Albania", "Andorra", "Unique_API"]}
        )

        matching_stats = identify_matching_countries(owid_df, api_df)

        # Assertions
        assert matching_stats["total_matched"] == 2  # Afghanistan, Albania
        assert matching_stats["owid_only_count"] == 2  # Algeria, Unique_OWID
        assert matching_stats["api_only_count"] == 2  # Andorra, Unique_API
        assert "Afghanistan" in matching_stats["matched_countries"]
        assert "Albania" in matching_stats["matched_countries"]
        assert "Algeria" in matching_stats["owid_only"]
        assert "Andorra" in matching_stats["api_only"]


class TestCountryMappings:
    """Test cases for country name mapping configurations."""

    def test_country_mapping_completeness(self):
        """Test that country mappings are reasonable."""
        # Check that mapping dictionary is not empty
        assert len(COUNTRY_NAME_MAPPING) > 0

        # Check some known mappings
        assert COUNTRY_NAME_MAPPING.get("Bosnia and Herzegovina") == "Bosnia"
        assert COUNTRY_NAME_MAPPING.get("Cape Verde") == "Cabo Verde"
        assert COUNTRY_NAME_MAPPING.get("United States") == "USA"

    def test_exclude_regions_completeness(self):
        """Test that exclude regions list covers common aggregates."""
        # Check that exclude list contains common non-countries
        assert "World" in EXCLUDE_REGIONS
        assert "Europe" in EXCLUDE_REGIONS
        assert "Africa" in EXCLUDE_REGIONS
        assert "Diamond Princess" in EXCLUDE_REGIONS


class TestIntegration:
    """Integration tests using realistic data structures."""

    def test_full_cleaning_pipeline(self):
        """Test the complete data cleaning pipeline."""
        # Create realistic mock data
        owid_df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Bosnia and Herzegovina", "World", "Albania"],
                "date": pd.date_range("2024-01-01", periods=4),
                "iso_code": ["AFG", "BIH", "OWID_WRL", "ALB"],
                "population": [40000000, 3300000, 7800000000, 3000000],
                "total_cases": [1000, 800, 50000000, 500],
                "total_deaths": [50, 40, 1000000, 25],
            }
        )

        api_df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Bosnia", "Albania", "Diamond Princess"],
                "iso_code": ["AFG", "BIH", "ALB", "XXX"],
                "population": [40000000, 3300000, 3000000, 3711],
                "current_cases": [1200, 900, 600, 712],
                "current_deaths": [60, 45, 30, 14],
            }
        )

        from covid_integration.data_cleaner import clean_all_data

        owid_clean, api_clean, quality_report = clean_all_data(owid_df, api_df)

        # Test that cleaning worked
        assert len(owid_clean) > 0
        assert len(api_clean) > 0
        assert "World" not in owid_clean["country"].values
        assert "Diamond Princess" not in api_clean["country"].values

        # Test that matching worked
        assert quality_report["country_matching"]["total_matched"] >= 2
        assert "owid_quality" in quality_report
        assert "api_quality" in quality_report


class TestDataMerger:
    """Test cases for data merger functionality."""

    def test_create_country_mapping_table(self):
        """Test creation of country mapping table."""
        owid_df = pd.DataFrame(
            {"country_standardized": ["Afghanistan", "Albania", "Algeria", "OWID_Only"]}
        )

        api_df = pd.DataFrame(
            {"country_standardized": ["Afghanistan", "Albania", "Andorra", "API_Only"]}
        )

        mapping_table = create_country_mapping_table(owid_df, api_df)

        # Check structure
        expected_columns = ["country", "in_owid", "in_api", "can_merge"]
        assert all(col in mapping_table.columns for col in expected_columns)

        # Check specific mappings
        afghanistan_row = mapping_table[mapping_table['country'] == 'Afghanistan'].iloc[0]
        assert afghanistan_row['in_owid']  # Should be truthy
        assert afghanistan_row['in_api']   # Should be truthy
        assert afghanistan_row['can_merge']  # Should be truthy
        
        owid_only_row = mapping_table[mapping_table['country'] == 'OWID_Only'].iloc[0]
        assert owid_only_row['in_owid']  # Should be truthy
        assert not owid_only_row['in_api']  # Should be falsy
        assert not owid_only_row['can_merge']  # Should be falsy
        
        api_only_row = mapping_table[mapping_table['country'] == 'API_Only'].iloc[0]
        assert not api_only_row['in_owid']  # Should be falsy
        assert api_only_row['in_api']  # Should be truthy
        assert not api_only_row['can_merge']  # Should be falsy

    def test_align_temporal_data(self):
        """Test temporal alignment of datasets."""
        # Create OWID data with multiple dates per country
        owid_df = pd.DataFrame(
            {
                "country_standardized": ["Afghanistan", "Afghanistan", "Albania", "Albania"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
                "total_cases": [1000, 1100, 500, 550],
                "total_deaths": [50, 55, 25, 27],
                "population": [40000000, 40000000, 3000000, 3000000],
                "iso_code": ["AFG", "AFG", "ALB", "ALB"],
                "country": ["Afghanistan", "Afghanistan", "Albania", "Albania"],
            }
        )

        api_df = pd.DataFrame(
            {
                "country_standardized": ["Afghanistan", "Albania"],
                "current_cases": [1200, 600],
                "current_deaths": [60, 30],
                "country": ["Afghanistan", "Albania"],
                "iso_code": ["AFG", "ALB"],
            }
        )

        owid_aligned, api_aligned = align_temporal_data(owid_df, api_df, "latest")

        # Check that we got latest data for each country
        assert len(owid_aligned) == 2  # One row per country
        assert "owid_date" in owid_aligned.columns
        assert "owid_total_cases" in owid_aligned.columns

        # Check that Afghanistan got the latest values (from 2024-01-02)
        afghanistan_row = owid_aligned[owid_aligned["country_standardized"] == "Afghanistan"].iloc[
            0
        ]
        assert afghanistan_row["owid_total_cases"] == 1100
        assert afghanistan_row["owid_date"] == pd.to_datetime("2024-01-02")

        # Check API data has proper prefixes
        assert "api_current_cases" in api_aligned.columns
        assert len(api_aligned) == 2

    def test_calculate_trend_metrics(self):
        """Test trend metrics calculation."""
        # Create data spanning 35 days with clear trends
        dates = pd.date_range("2024-01-01", periods=35, freq="D")
        owid_df = pd.DataFrame(
            {
                "country_standardized": ["TestCountry"] * 35,
                "date": dates,
                "total_cases": range(1000, 1035),  # Increasing by 1 each day
                "total_deaths": range(50, 85),  # Increasing by 1 each day
                "new_cases": [1] * 35,  # Consistent 1 new case per day
                "new_deaths": [1] * 35,  # Consistent 1 new death per day
            }
        )

        trend_df = calculate_trend_metrics(owid_df, window_days=30)

        # Should have one row for TestCountry
        assert len(trend_df) == 1
        test_row = trend_df.iloc[0]

        # Check calculated metrics
        assert test_row["country_standardized"] == "TestCountry"
        assert test_row["trend_window_days"] == 30
        assert test_row["cases_change_absolute"] > 0  # Should show increase
        assert test_row["avg_daily_new_cases"] == 1.0
        assert test_row["avg_daily_new_deaths"] == 1.0
        assert test_row["data_points_available"] >= 30

    def test_calculate_derived_metrics(self):
        """Test calculation of derived metrics."""
        merged_df = pd.DataFrame(
            {
                "country_standardized": ["Afghanistan", "Albania"],
                "owid_total_cases": [1000.0, 500.0],
                "api_current_cases": [1200, 600],
                "owid_total_deaths": [50.0, 25.0],
                "api_current_deaths": [60, 30],
                "owid_population": [40000000, 3000000],
            }
        )

        result_df = calculate_derived_metrics(merged_df)

        # Check that new columns were added
        expected_columns = [
            "cases_data_gap",
            "cases_data_gap_percent",
            "deaths_data_gap",
            "deaths_data_gap_percent",
            "current_cases_per_100k",
            "current_deaths_per_100k",
            "current_case_fatality_rate",
        ]

        for col in expected_columns:
            assert col in result_df.columns

        # Check specific calculations for Afghanistan
        afghanistan_row = result_df[result_df["country_standardized"] == "Afghanistan"].iloc[0]
        assert afghanistan_row["cases_data_gap"] == 200  # 1200 - 1000
        assert afghanistan_row["cases_data_gap_percent"] == 20.0  # 200/1000 * 100
        assert afghanistan_row["current_cases_per_100k"] == 3.0  # 1200/40M * 100k
        assert afghanistan_row["current_case_fatality_rate"] == 5.0  # 60/1200 * 100

    def test_generate_integration_summary(self):
        """Test integration summary generation."""
        merged_df = pd.DataFrame(
            {
                "country_standardized": ["Afghanistan", "Albania", "Algeria"],
                "owid_total_cases": [1000.0, 500.0, np.nan],
                "api_current_cases": [1200, 600, 800],
                "owid_total_deaths": [50.0, 25.0, 40.0],
                "api_current_deaths": [60, 30, 45],
                "cases_data_gap_percent": [20.0, 20.0, np.nan],
                "owid_data_age_days": [300, 310, 320],
                "avg_daily_new_cases": [1.0, 0.5, np.nan],
                "current_cases_per_100k": [3.0, 20.0, 18.0],
            }
        )

        summary = generate_integration_summary(merged_df)

        # Check structure
        assert "integration_timestamp" in summary
        assert "total_countries_merged" in summary
        assert summary["total_countries_merged"] == 3

        # Check data completeness
        assert "data_completeness" in summary
        owid_cases_completeness = summary["data_completeness"]["owid_total_cases"]
        assert owid_cases_completeness["available"] == 2
        assert owid_cases_completeness["missing"] == 1
        assert owid_cases_completeness["completeness_percent"] == 66.7

        # Check quality flags
        assert "data_quality_flags" in summary
        assert "temporal_analysis" in summary

    def test_merge_datasets_basic(self):
        """Test basic dataset merging functionality."""
        # Create minimal test data
        owid_df = pd.DataFrame(
            {
                "country_standardized": ["Afghanistan", "Albania"],
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "total_cases": [1000.0, 500.0],
                "total_deaths": [50.0, 25.0],
                "population": [40000000, 3000000],
                "iso_code": ["AFG", "ALB"],
                "country": ["Afghanistan", "Albania"],
            }
        )

        api_df = pd.DataFrame(
            {
                "country_standardized": ["Afghanistan", "Albania"],
                "current_cases": [1200, 600],
                "current_deaths": [60, 30],
                "country": ["Afghanistan", "Albania"],
                "iso_code": ["AFG", "ALB"],
            }
        )

        merged_df = merge_datasets(owid_df, api_df, include_trends=False)

        # Basic checks
        assert len(merged_df) == 2
        assert "country_standardized" in merged_df.columns
        assert "owid_total_cases" in merged_df.columns
        assert "api_current_cases" in merged_df.columns
        assert "merge_timestamp" in merged_df.columns

        # Check that derived metrics were calculated
        assert "cases_data_gap" in merged_df.columns
        assert "current_cases_per_100k" in merged_df.columns


class TestDataMergerIntegration:
    """Integration tests for the complete merger pipeline."""

    def test_full_integration_pipeline(self):
        """Test the complete integration pipeline with realistic data."""
        # Create realistic OWID data with multiple dates to test temporal alignment
        owid_df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Afghanistan", "Albania", "Albania", "World"],
                "country_standardized": [
                    "Afghanistan",
                    "Afghanistan",
                    "Albania",
                    "Albania",
                    "World",
                ],
                "date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"]
                ),
                "iso_code": ["AFG", "AFG", "ALB", "ALB", "OWID_WRL"],
                "population": [40000000, 40000000, 3000000, 3000000, 8000000000],
                "total_cases": [1000.0, 1100.0, 500.0, 550.0, 50000000.0],
                "total_deaths": [50.0, 55.0, 25.0, 27.0, 1000000.0],
                "data_source": ["owid_historical"] * 5,
            }
        )

        # Create realistic API data
        api_df = pd.DataFrame(
            {
                "country": ["Afghanistan", "Albania"],
                "country_standardized": ["Afghanistan", "Albania"],
                "iso_code": ["AFG", "ALB"],
                "population": [40000000, 3000000],
                "current_cases": [1200, 600],
                "current_deaths": [60, 30],
                "cases_per_100k": [3.0, 20.0],
                "data_source": ["disease_sh_current"] * 2,
            }
        )

        # Run integration
        merged_df, summary = integrate_covid_data(owid_df, api_df)

        # Check results
        assert len(merged_df) >= 1  # Should have some merged countries
        assert summary["total_countries_merged"] >= 1
        assert "data_completeness" in summary
        assert "integration_timestamp" in summary

        # Check that World aggregate was filtered out in the process
        assert (
            "World" not in merged_df["country_standardized"].values if len(merged_df) > 0 else True
        )

        # Check that we have the expected countries (Afghanistan, Albania should merge)
        if len(merged_df) > 0:
            merged_countries = set(merged_df["country_standardized"].values)
            assert "Afghanistan" in merged_countries or "Albania" in merged_countries


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
