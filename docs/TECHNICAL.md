# Technical Documentation

## Architecture

### System Overview

This project implements a complete data engineering pipeline that integrates multiple COVID-19 data sources, validates data quality, and presents insights through an interactive dashboard.

```
┌─────────────┐      ┌─────────────┐
│  OWID CSV   │      │ disease.sh  │
│  (Historical)│      │    API      │
└──────┬──────┘      └──────┬──────┘
       │                    │
       └────────┬───────────┘
                │
         ┌──────▼──────┐
         │ Data Loader │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │Data Cleaner │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │ Data Merger │
         └──────┬──────┘
                │
         ┌──────▼──────┐
         │  Streamlit  │
         │  Dashboard  │
         └─────────────┘
```

### Technical Stack

- **Language**: Python 3.9+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Testing**: pytest
- **Code Quality**: black, isort, flake8
- **Dependency Management**: Poetry

## Data Pipeline

### 1. Data Loading (`data_loader.py`)

**Sources:**
- **OWID CSV**: Historical time series data (429K+ rows, 67 columns)
- **disease.sh API**: Current country-level snapshots (231 countries)

**Process:**
- Downloads OWID data from GitHub
- Fetches disease.sh API data
- Performs initial validation
- Selects relevant columns

**Output:** Raw dataframes for cleaning

### 2. Data Cleaning (`data_cleaner.py`)

**Standardization:**
- Country name harmonization (25+ mapping rules)
- Filters non-country aggregates (continents, income groups)
- Handles missing values with forward fill
- Calculates derived metrics

**Quality Checks:**
- Missing data analysis
- Population threshold validation
- Completeness scoring

**Output:** Cleaned, standardized dataframes

### 3. Data Integration (`data_merger.py`)

**Temporal Alignment:**
- Extracts latest OWID data point per country
- Aligns with API snapshot
- Calculates data gaps

**Derived Metrics:**
- Cases per 100k population
- Deaths per 100k population
- Case fatality rates
- 30-day trend calculations

**Quality Flags:**
- Large data gaps (>10%)
- Old data (>90 days)
- Population anomalies

**Output:** Merged dataset (194 countries)

### 4. Visualization (`visualizer.py`)

**Static Plots:**
- Top countries rankings
- Comparative analysis
- Data quality heatmaps
- Trend visualizations

**Interactive Dashboard:**
- Country comparisons
- World maps
- Searchable raw data
- Quality dashboards

## Data Sources & Methodology

### Our World in Data (OWID)

**URL:** https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/owid-covid-data.csv

**Coverage:**
- Time series from 2020-01-01 to 2024-08-14
- 255 locations (countries + aggregates)
- 67 columns of metrics

**Key Columns Used:**
- `location`, `date`, `iso_code`, `population`
- `total_cases`, `new_cases`, `total_deaths`, `new_deaths`
- `total_tests`, `people_vaccinated`, `people_fully_vaccinated`

### disease.sh API

**URL:** https://disease.sh/v3/covid-19/countries

**Coverage:**
- Current snapshot data
- 231 countries
- Updated regularly

**Key Fields Used:**
- `country`, `cases`, `deaths`, `population`
- `tests`, `active`, `critical`, `recovered`

### Country Name Harmonization

**Challenge:** Different naming conventions between sources
- OWID: "United States"
- API: "USA"

**Solution:** 25+ mapping rules in `config/constants.py`

**Examples:**
```python
COUNTRY_NAME_MAPPING = {
    "United States": "USA",
    "United Kingdom": "UK",
    "South Korea": "S. Korea",
    "Czech Republic": "Czechia",
    # ... 21 more mappings
}
```

**Result:** 194 countries successfully matched (80%+ match rate)

## Integration Process

### Step-by-Step

1. **Load OWID Data**
   - Download CSV
   - Parse 429K rows
   - Filter to key columns

2. **Load API Data**
   - Fetch JSON
   - Parse 231 countries
   - Extract metrics

3. **Clean OWID**
   - Standardize country names
   - Filter aggregates
   - Handle missing values
   - Calculate per 100k metrics

4. **Clean API**
   - Standardize country names
   - Validate population data
   - Remove non-countries

5. **Temporal Alignment**
   - Get latest OWID date per country
   - Match with API snapshot
   - Calculate time gaps

6. **Merge Datasets**
   - Inner join on country name
   - Combine OWID + API columns
   - Calculate data gaps

7. **Derive Metrics**
   - Population-adjusted rates
   - Case fatality rates
   - Trend calculations (30-day)

8. **Quality Assessment**
   - Flag large data gaps
   - Flag old data
   - Flag population anomalies

### Data Quality Flags

**Large Data Gap** (>10% difference):
```python
if abs(api_cases - owid_cases) / owid_cases > 0.10:
    flag_large_gap()
```

**Old Data** (>90 days):
```python
if (today - latest_date).days > 90:
    flag_old_data()
```

**Population Anomaly**:
```python
if not (1000 < population < 2e9):
    flag_population_anomaly()
```

## Key Achievements

- ✅ **194 countries integrated** (80%+ match rate)
- ✅ **Comprehensive data quality transparency**
- ✅ **Production-ready error handling and logging**
- ✅ **21+ integration tests** covering all major functionality
- ✅ **Modular, maintainable architecture**

## Performance Considerations

- **CSV Download**: ~5-7 seconds
- **API Fetch**: ~1-2 seconds
- **Data Processing**: <1 second
- **Total Pipeline**: ~8-10 seconds
- **Dashboard Load**: ~10-15 seconds

## Error Handling

All modules implement comprehensive error handling:
- Network errors (requests.RequestException)
- Parse errors (json.JSONDecodeError, pd.errors)
- Validation errors (custom checks)
- Informative logging at each step

## Testing Strategy

**Integration Tests:** `tests/test_integration.py`
- 21+ tests covering all major functions
- Mock data for fast execution
- Edge cases and error conditions

**Test Categories:**
- Data loading (OWID, API)
- Country standardization
- Data filtering
- Data merging
- Metric calculations
- Quality validation

## Limitations

⚠️ **Historical Data Only:**
- OWID data ends August 2024
- API data is from August 2024 snapshot
- No real-time updates

⚠️ **Country Matching:**
- Some countries don't match between sources
- Different naming conventions
- Some territories excluded

⚠️ **Data Gaps:**
- Missing data for some metrics
- Different update frequencies
- Reporting delays vary by country
