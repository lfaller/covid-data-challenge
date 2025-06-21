# COVID-19 Multi-source Data Integration Challenge

## Problem Statement
**Scenario:** A public health research team needs to analyze the relationship between COVID-19 case trends, testing capacity, and vaccination rollout across different countries. The data exists in multiple formats from different authoritative sources that need to be integrated for comprehensive analysis.

## Data Sources

### CSV Source: Johns Hopkins COVID-19 Case Data
- **URL:** https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
- **Format:** Time series with countries as rows, dates as columns
- **Contains:** Cumulative confirmed cases by country/region over time

### JSON Source: Our World in Data COVID-19 Testing & Vaccination
- **URL:** https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.json
- **Format:** Nested JSON with country codes as keys
- **Contains:** Daily testing rates, vaccination data, population info

## Real-World Integration Challenges

1. **Country Name Mismatches:**
   - JHU uses "US", OWID uses "United States"
   - JHU has "Korea, South", OWID has "South Korea"
   - Some countries have different spellings/formats

2. **Date Format Differences:**
   - JHU: Column headers like "1/22/20"
   - OWID: ISO format "2020-01-22"

3. **Data Granularity:**
   - JHU: Some countries broken down by province/state
   - OWID: Country-level only

4. **Missing Data:**
   - Not all countries have testing data
   - Vaccination data starts later than case data
   - Some time periods have gaps

## Implementation Plan

### Step 1: Data Loading & Parsing
```python
def load_jhu_data(url):
    """Load and reshape JHU time series data"""
    
def load_owid_data(url):
    """Load and parse OWID JSON data"""
```

### Step 2: Data Cleaning & Standardization
```python
def standardize_country_names(df):
    """Create mapping for country name discrepancies"""
    
def convert_date_formats(df):
    """Standardize date formats across sources"""
```

### Step 3: Data Integration
```python
def merge_datasets(jhu_df, owid_df):
    """Intelligently merge datasets handling missing data"""
```

### Step 4: Quality Assessment
```python
def generate_quality_report(merged_df):
    """Assess data completeness and identify issues"""
```

### Step 5: Output Generation
- Clean merged dataset (CSV)
- Data quality report (text/markdown)
- Summary statistics

## Expected Deliverables

1. **`covid_integration.py`** - Main script
2. **`README.md`** - Problem description and usage
3. **`requirements.txt`** - Dependencies
4. **Output files:**
   - `merged_covid_data.csv` - Integrated dataset
   - `data_quality_report.txt` - Quality assessment
   - `summary_stats.txt` - Key findings

## Success Metrics
- Successfully merge data for 50+ countries
- Handle at least 5 country name mismatches
- Identify and report missing data patterns
- Generate actionable quality metrics

## Technical Approach
- Use pandas for data manipulation
- Implement robust error handling
- Add logging for transparency
- Include basic unit tests for key functions