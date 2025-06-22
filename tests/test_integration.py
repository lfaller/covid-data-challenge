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

# Import modules to test
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
from covid_integration.data_loader import (
    compare_data_sources,
    load_disease_sh_data,
    parse_disease_sh_json,
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


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
