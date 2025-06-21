"""
COVID-19 Data Loading Module

This module handles loading and initial parsing of COVID-19 data from multiple sources:
- Our World in Data (OWID) comprehensive historical dataset (CSV format)
- disease.sh API for current country-level data (JSON format)
"""

import pandas as pd
import requests
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data source URLs
OWID_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/owid-covid-data.csv"
DISEASE_SH_API_URL = "https://disease.sh/v3/covid-19/countries"


def load_owid_data(url: str = OWID_CSV_URL) -> pd.DataFrame:
    """
    Load Our World in Data COVID-19 comprehensive historical dataset.
    
    Args:
        url: URL to the OWID CSV file
        
    Returns:
        DataFrame with historical COVID-19 metrics by country and date
        
    Raises:
        requests.RequestException: If data download fails
        pd.errors.ParserError: If CSV parsing fails
    """
    try:
        logger.info(f"Loading OWID historical data from {url}")
        
        # Download and read CSV directly into DataFrame
        df = pd.read_csv(url)
        logger.info(f"Loaded OWID data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display basic info about the data structure
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Countries: {df['location'].nunique()}")
        logger.info(f"Key columns: {list(df.columns[:15])}...")  # First 15 columns
        
        # Clean and standardize the data
        df_clean = clean_owid_data(df)
        
        logger.info(f"Cleaned OWID data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean
        
    except requests.RequestException as e:
        logger.error(f"Failed to download OWID data: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to process OWID data: {e}")
        raise


def clean_owid_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize OWID data.
    
    Args:
        df: Raw OWID DataFrame
        
    Returns:
        Cleaned DataFrame with standardized columns
    """
    # Make a copy to avoid modifying original
    df_clean = df.copy()
    
    # Rename location to country for consistency
    df_clean.rename(columns={'location': 'country'}, inplace=True)
    
    # Convert date column to datetime
    df_clean['date'] = pd.to_datetime(df_clean['date'])
    
    # Select key columns we want to work with
    key_columns = [
        'country', 'date', 'iso_code', 'population',
        'total_cases', 'new_cases', 'total_deaths', 'new_deaths',
        'total_tests', 'new_tests', 'people_vaccinated', 
        'people_fully_vaccinated', 'total_vaccinations',
        'tests_per_case', 'positive_rate'
    ]
    
    # Only keep columns that exist in the dataset
    available_columns = [col for col in key_columns if col in df_clean.columns]
    df_clean = df_clean[available_columns]
    
    # Remove rows for world data and other aggregates
    exclude_codes = ['OWID_WRL', 'OWID_HIC', 'OWID_LIC', 'OWID_LMC', 'OWID_UMC']
    df_clean = df_clean[~df_clean['iso_code'].isin(exclude_codes)]
    
    # Sort by country and date
    df_clean = df_clean.sort_values(['country', 'date']).reset_index(drop=True)
    
    return df_clean


def load_disease_sh_data(url: str = DISEASE_SH_API_URL) -> pd.DataFrame:
    """
    Load current COVID-19 data from disease.sh API.
    
    Args:
        url: URL to the disease.sh API endpoint
        
    Returns:
        DataFrame with current COVID-19 metrics by country
        
    Raises:
        requests.RequestException: If API request fails
        json.JSONDecodeError: If JSON parsing fails
    """
    try:
        logger.info(f"Loading current data from disease.sh API: {url}")
        
        # Make API request
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse JSON response
        data = response.json()
        logger.info(f"Loaded disease.sh data for {len(data)} countries")
        
        # Convert to DataFrame
        df_api = parse_disease_sh_json(data)
        
        logger.info(f"Parsed disease.sh data: {df_api.shape[0]} rows, {df_api.shape[1]} columns")
        return df_api
        
    except requests.RequestException as e:
        logger.error(f"Failed to fetch disease.sh API data: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse disease.sh JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to process disease.sh data: {e}")
        raise


def parse_disease_sh_json(data: List[Dict]) -> pd.DataFrame:
    """
    Parse disease.sh JSON API response into a DataFrame.
    
    Args:
        data: List of country dictionaries from disease.sh API
        
    Returns:
        DataFrame with current metrics by country
    """
    rows = []
    
    for country_record in data:
        # Extract country info
        country_info = country_record.get('countryInfo', {})
        
        row = {
            'country': country_record.get('country'),
            'iso_code': country_info.get('iso3'),  # 3-letter ISO code
            'iso2_code': country_info.get('iso2'), # 2-letter ISO code
            'country_id': country_info.get('_id'),
            'latitude': country_info.get('lat'),
            'longitude': country_info.get('long'),
            'flag_url': country_info.get('flag'),
            'population': country_record.get('population'),
            'current_cases': country_record.get('cases'),
            'current_deaths': country_record.get('deaths'),
            'current_recovered': country_record.get('recovered'),
            'current_active': country_record.get('active'),
            'current_critical': country_record.get('critical'),
            'cases_per_million': country_record.get('casesPerOneMillion'),
            'deaths_per_million': country_record.get('deathsPerOneMillion'),
            'tests_total': country_record.get('tests'),
            'tests_per_million': country_record.get('testsPerOneMillion'),
            'today_cases': country_record.get('todayCases'),
            'today_deaths': country_record.get('todayDeaths'),
            'today_recovered': country_record.get('todayRecovered'),
            'updated_timestamp': country_record.get('updated')
        }
        rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Convert timestamp to datetime
    if 'updated_timestamp' in df.columns:
        df['last_updated'] = pd.to_datetime(df['updated_timestamp'], unit='ms')
        df.drop('updated_timestamp', axis=1, inplace=True)
    
    # Sort by country name
    df = df.sort_values('country').reset_index(drop=True)
    
    return df


def compare_data_sources(owid_df: pd.DataFrame, api_df: pd.DataFrame) -> Dict:
    """
    Compare the two data sources to identify integration challenges.
    
    Args:
        owid_df: OWID historical DataFrame
        api_df: disease.sh current DataFrame
        
    Returns:
        Dictionary with comparison statistics
    """
    # Get latest date from OWID data
    latest_owid_date = owid_df['date'].max()
    
    # Get unique countries from each source
    owid_countries = set(owid_df['country'].unique())
    api_countries = set(api_df['country'].unique())
    
    # Find overlaps and differences
    common_countries = owid_countries.intersection(api_countries)
    owid_only = owid_countries - api_countries
    api_only = api_countries - owid_countries
    
    comparison_stats = {
        'owid_total_countries': len(owid_countries),
        'api_total_countries': len(api_countries),
        'common_countries': len(common_countries),
        'owid_only_count': len(owid_only),
        'api_only_count': len(api_only),
        'latest_owid_date': latest_owid_date,
        'owid_only_countries': sorted(list(owid_only))[:10],  # First 10
        'api_only_countries': sorted(list(api_only))[:10],    # First 10
        'common_countries_sample': sorted(list(common_countries))[:10]  # First 10
    }
    
    return comparison_stats


def load_all_data() -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Load both OWID historical and disease.sh current datasets.
    
    Returns:
        Tuple of (owid_dataframe, disease_sh_dataframe, comparison_stats)
    """
    logger.info("Starting data loading process...")
    
    # Load OWID historical data
    owid_df = load_owid_data()
    
    # Load disease.sh current data  
    api_df = load_disease_sh_data()
    
    # Compare the data sources
    comparison_stats = compare_data_sources(owid_df, api_df)
    
    logger.info("Data loading completed successfully!")
    
    # Display comparison stats
    logger.info(f"OWID data: {comparison_stats['owid_total_countries']} countries, "
                f"latest date: {comparison_stats['latest_owid_date']}")
    logger.info(f"disease.sh API: {comparison_stats['api_total_countries']} countries")
    logger.info(f"Common countries: {comparison_stats['common_countries']}")
    logger.info(f"Countries only in OWID: {comparison_stats['owid_only_count']}")
    logger.info(f"Countries only in API: {comparison_stats['api_only_count']}")
    
    return owid_df, api_df, comparison_stats


if __name__ == "__main__":
    # Test the data loading functions
    try:
        owid_data, api_data, stats = load_all_data()
        
        print("\n=== OWID Historical Data Sample ===")
        print(owid_data.head())
        print(f"\nOWID Data Info:")
        print(owid_data.info())
        
        print("\n=== disease.sh Current Data Sample ===")
        print(api_data.head())
        print(f"\ndisease.sh Data Info:")
        print(api_data.info())
        
        print("\n=== Data Source Comparison ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        raise