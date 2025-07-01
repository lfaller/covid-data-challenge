"""
COVID-19 Data Integration - Configuration Constants

Centralized configuration and constants for the entire project.
This module contains all hardcoded values, mappings, and configuration
parameters used across different modules.
"""

# Data source URLs
OWID_CSV_URL = "https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/owid-covid-data.csv"
DISEASE_SH_API_URL = "https://disease.sh/v3/covid-19/countries"

# Country name mapping dictionary to handle mismatches between data sources
COUNTRY_NAME_MAPPING = {
    # OWID name -> disease.sh API name
    "Bosnia and Herzegovina": "Bosnia",
    "Cape Verde": "Cabo Verde",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Democratic Republic of Congo": "DRC",
    "East Timor": "Timor-Leste",
    "Curacao": "Curaçao",
    "Bonaire Sint Eustatius and Saba": "Caribbean Netherlands",
    "United States": "USA",
    "United Kingdom": "UK",
    "South Korea": "S. Korea",
    "Czech Republic": "Czechia",
    "North Macedonia": "Macedonia",
    "Myanmar": "Burma",
    "Republic of the Congo": "Congo",
    "Eswatini": "Swaziland",
    "Vatican": "Holy See (Vatican City State)",
    "Brunei": "Brunei Darussalam",
    "Moldova": "Moldova, Republic of",
    "Russia": "Russian Federation",
    "Syria": "Syrian Arab Republic",
    "Tanzania": "Tanzania, United Republic of",
    "Turkey": "Turkey",
    "Venezuela": "Venezuela, Bolivarian Republic of",
    "Vietnam": "Viet Nam",
    "Laos": "Lao People's Democratic Republic",
}

# Countries/regions to exclude (aggregates, non-countries)
EXCLUDE_REGIONS = {
    "World",
    "Africa",
    "Asia",
    "Europe",
    "North America",
    "South America",
    "Oceania",
    "European Union",
    "High income",
    "Low income",
    "Lower middle income",
    "Upper middle income",
    "OECD countries",
    "International",
    "MS Zaandam",
    "Diamond Princess",
}

# Default parameters for analysis
DEFAULT_TREND_WINDOW_DAYS = 30
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_TIMEOUT_SECONDS = 30

# Data quality thresholds
MIN_POPULATION_THRESHOLD = 1000
MAX_POPULATION_THRESHOLD = 2e9
LARGE_DATA_GAP_THRESHOLD = 10.0  # percent
OLD_DATA_THRESHOLD_DAYS = 90

# Visualization constants
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_DPI = 300
DEFAULT_TOP_N_COUNTRIES = 15

# Color palette for consistent styling
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "accent": "#F18F01",
    "success": "#C73E1D",
    "warning": "#F4A261",
    "info": "#264653",
}

# Column name mappings for consistency
COLUMN_MAPPINGS = {
    # Standard column names used throughout the project
    "country_col": "country_standardized",
    "date_col": "date",
    "iso_col": "iso_code",
    "population_col": "population",
    "cases_col": "total_cases",
    "deaths_col": "total_deaths",
}

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# File extensions and formats
SUPPORTED_OUTPUT_FORMATS = [".csv", ".json", ".xlsx"]
DEFAULT_IMAGE_FORMAT = "png"

# API and data processing limits
MAX_RETRIES = 3
BATCH_SIZE = 1000
MAX_COUNTRIES_FOR_COMPARISON = 10

# Data validation rules
REQUIRED_OWID_COLUMNS = ["country", "date", "iso_code", "total_cases"]
REQUIRED_API_COLUMNS = ["country", "cases", "deaths", "population"]

# Cache and performance settings
ENABLE_CACHING = True
CACHE_EXPIRY_HOURS = 24
