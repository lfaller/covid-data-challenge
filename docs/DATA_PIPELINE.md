# Data Pipeline: Load, Clean, and Merge

This document explains the complete data processing pipeline that integrates COVID-19 data from multiple sources.

## Pipeline Overview

```
┌─────────────────┐      ┌──────────────────┐
│   OWID CSV      │      │  disease.sh API  │
│   (Historical)  │      │   (Current)      │
│  429K+ rows     │      │  231 countries   │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
    ┌────────────────────────────────┐
    │   STEP 1: DATA LOADING         │
    │   • Download & parse           │
    │   • Initial validation         │
    │   • Select key columns         │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │   STEP 2: DATA CLEANING        │
    │   • Standardize names          │
    │   • Filter non-countries       │
    │   • Handle missing values      │
    │   • Validate quality           │
    └────────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────────┐
    │   STEP 3: DATA MERGING         │
    │   • Temporal alignment         │
    │   • Country matching           │
    │   • Calculate derived metrics  │
    │   • Quality flagging           │
    └────────────┬───────────────────┘
                 │
                 ▼
         ┌───────────────┐
         │ Final Dataset │
         │ 194 countries │
         └───────────────┘
```

---

## STEP 1: Data Loading

### Module: `data_loader.py`

### 1.1 Load OWID Historical Data

**Function:** `load_owid_data()`

**Source:** Our World in Data (OWID)
- **URL:** https://raw.githubusercontent.com/owid/covid-19-data/refs/heads/master/public/data/owid-covid-data.csv
- **Format:** CSV
- **Size:** ~429,000 rows × 67 columns
- **Coverage:** 2020-01-01 to 2024-08-14
- **Locations:** 255 (countries + aggregates)

**Process:**
1. Download CSV from GitHub
2. Parse into pandas DataFrame
3. Log basic statistics (rows, columns, date range)
4. Call `clean_owid_data()` for preprocessing
5. Return cleaned DataFrame

**Key Columns Selected:**
```python
[
    'country',          # Renamed from 'location'
    'date',            # Datetime format
    'iso_code',        # ISO 3166-1 alpha-3 code
    'population',      # Total population
    'total_cases',     # Cumulative cases
    'new_cases',       # Daily new cases
    'total_deaths',    # Cumulative deaths
    'new_deaths',      # Daily new deaths
    'total_tests',     # Cumulative tests
    'new_tests',       # Daily new tests
    'people_vaccinated',
    'people_fully_vaccinated',
    'total_vaccinations',
    'tests_per_case',
    'positive_rate'
]
```

**Cleaning Steps in `clean_owid_data()`:**
```python
1. Rename 'location' → 'country'
2. Convert date to datetime
3. Select only key columns (reduce from 67 to 15)
4. Filter out aggregate rows (OWID_WRL, OWID_HIC, etc.)
5. Sort by country and date
```

**Output:**
- ~416,000 rows × 15 columns
- 250 countries (after removing aggregates)
- Ready for standardization

---

### 1.2 Load disease.sh API Data

**Function:** `load_disease_sh_data()`

**Source:** disease.sh API
- **URL:** https://disease.sh/v3/covid-19/countries
- **Format:** JSON
- **Size:** 231 countries
- **Coverage:** Current snapshot (August 2024)

**Process:**
1. Fetch JSON data from API
2. Parse into Python list of dictionaries
3. Call `parse_disease_sh_json()` to convert to DataFrame
4. Return structured DataFrame

**API Response Fields:**
```json
{
    "country": "USA",
    "countryInfo": {
        "iso2": "US",
        "iso3": "USA"
    },
    "cases": 103436829,
    "todayCases": 0,
    "deaths": 1127152,
    "todayDeaths": 0,
    "recovered": 101270669,
    "active": 1039008,
    "critical": 1644,
    "tests": 1174101620,
    "population": 334998398
}
```

**Parsing in `parse_disease_sh_json()`:**
```python
1. Extract country name
2. Extract ISO codes (iso2, iso3)
3. Extract metrics (cases, deaths, tests, population)
4. Calculate derived metrics:
   - cases_per_million
   - deaths_per_million
   - tests_per_million
5. Create DataFrame
```

**Output:**
- 231 rows × 21 columns
- Current snapshot data
- Ready for standardization

---

## STEP 2: Data Cleaning

### Module: `data_cleaner.py`

### 2.1 Standardize Country Names

**Function:** `standardize_country_names()`

**Problem:** Different sources use different country names
- OWID: "United States"
- API: "USA"

**Solution:** Country name mapping dictionary

**Mapping Examples:**
```python
COUNTRY_NAME_MAPPING = {
    "United States": "USA",
    "United Kingdom": "UK",
    "South Korea": "S. Korea",
    "Czech Republic": "Czechia",
    "Bosnia and Herzegovina": "Bosnia",
    "Cape Verde": "Cabo Verde",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Democratic Republic of Congo": "DRC",
    "Myanmar": "Burma",
    # ... 25+ total mappings
}
```

**Process for OWID Data:**
```python
1. Create 'country_standardized' column
2. Apply mapping: OWID name → API name
3. Keep original name if no mapping exists
4. Log number of mappings applied
```

**Process for API Data:**
```python
1. Create 'country_standardized' column
2. Use original API name (no mapping)
3. API names are already standardized
```

**Output:**
- Both datasets have 'country_standardized' column
- Enables matching across sources
- Original 'country' column preserved

---

### 2.2 Filter Valid Countries

**Function:** `filter_valid_countries()`
(now in `utils/data_filters.py`)

**Problem:** Datasets include non-country aggregates
- Continents (Africa, Asia, Europe)
- Income groups (High income, Low income)
- Organizations (OECD countries, European Union)

**Filter Criteria:**

**1. Exclude by Name:**
```python
EXCLUDE_REGIONS = {
    "World",
    "Africa", "Asia", "Europe", "North America", "South America", "Oceania",
    "European Union",
    "High income", "Low income", "Lower middle income", "Upper middle income",
    "OECD countries",
    "International",
    "MS Zaandam",        # Cruise ships
    "Diamond Princess"
}
```

**2. Exclude by Pattern:**
```python
# Remove entries containing these patterns
patterns = r"income|OECD|Union|International"
```

**3. Exclude by ISO Code:**
```python
# OWID aggregate codes
exclude_codes = ["OWID_WRL", "OWID_HIC", "OWID_LIC", "OWID_LMC", "OWID_UMC"]
```

**Process:**
```python
1. Count initial countries
2. Filter out EXCLUDE_REGIONS
3. Filter out pattern matches
4. Filter out aggregate ISO codes
5. Count final countries
6. Log: "Filtered countries: 255 → 250 (removed 5)"
```

**Output:**
- Only actual countries remain
- Aggregates removed
- Ready for integration

---

### 2.3 Validate Data Quality

**Function:** `validate_data_quality()`

**Purpose:** Assess data quality before merging

**Quality Checks:**

**1. Basic Statistics:**
```python
- Total rows
- Total countries
- Date range (min, max, span in days)
```

**2. Missing Data Analysis:**
```python
For each numeric column:
  - Count missing values
  - Calculate missing percentage
  - Flag if > threshold
```

**3. Population Outliers:**
```python
Check for unreasonable population values:
  - Too small: < 1,000
  - Too large: > 2 billion
Flag outliers for review
```

**4. Negative Values:**
```python
Check case/death columns for negative values
(Negative values indicate data errors)
```

**Output:**
```python
{
    "total_rows": 416005,
    "total_countries": 250,
    "date_range": ("2020-01-01", "2024-08-14"),
    "date_span_days": 1687,
    "missing_data": {
        "total_cases": {"missing_count": 5420, "missing_percentage": 1.30},
        "total_deaths": {"missing_count": 7891, "missing_percentage": 1.90},
        ...
    },
    "population_outliers": 2,
    "negative_values": 0
}
```

---

### 2.4 Clean OWID Data

**Function:** `clean_owid_data()` (in data_cleaner.py)

**Complete Process:**
```python
1. Standardize country names
   → Add 'country_standardized' column

2. Filter to valid countries
   → Remove aggregates and non-countries

3. Handle missing values
   → Forward fill cumulative columns within each country

4. Calculate derived metrics
   → Cases per 100k population
   → Deaths per 100k population
   → Case fatality rate

5. Validate data quality
   → Run quality checks
   → Log results

6. Sort and reset index
   → Sort by country and date
   → Reset index
```

**Missing Value Handling:**

For **cumulative columns** (total_cases, total_deaths, etc.):
```python
# Forward fill within each country
df.groupby('country')['total_cases'].ffill()
```

For **new/daily columns** (new_cases, new_deaths):
```python
# Calculate from cumulative if missing
df['new_cases'] = df.groupby('country')['total_cases'].diff()
df['new_cases'] = df['new_cases'].fillna(0).clip(lower=0)
```

**Derived Metrics Calculation:**
```python
# Cases per 100k population
df['cases_per_100k'] = (df['total_cases'] / df['population'] * 100000).round(2)

# Deaths per 100k population
df['deaths_per_100k'] = (df['total_deaths'] / df['population'] * 100000).round(2)

# Case fatality rate (%)
df['case_fatality_rate'] = np.where(
    df['total_cases'] > 0,
    (df['total_deaths'] / df['total_cases'] * 100).round(3),
    0
)
```

**Output:**
- Clean OWID dataset ready for merging
- Standardized country names
- Missing values handled
- Derived metrics calculated

---

### 2.5 Clean API Data

**Function:** `clean_api_data()` (in data_cleaner.py)

**Process:**
```python
1. Standardize country names
   → Add 'country_standardized' column (use original API names)

2. Filter to valid countries
   → Remove any non-countries

3. Validate population data
   → Check for outliers
   → Flag suspicious values

4. Calculate per-capita metrics
   → Cases per 100k
   → Deaths per 100k
   → Tests per 100k

5. Validate data quality
   → Run quality checks
   → Log results
```

**Output:**
- Clean API dataset ready for merging
- Standardized country names
- Quality validated

---

## STEP 3: Data Merging

### Module: `data_merger.py`

### 3.1 Temporal Alignment

**Function:** `align_temporal_data()`

**Challenge:** Different data frequencies
- OWID: Time series (daily data)
- API: Single snapshot (current)

**Solution:** Extract latest OWID data per country

**Process:**
```python
1. Group OWID data by country
2. Get latest date for each country
3. Extract last values:
   - date
   - total_cases
   - total_deaths
   - population
   - iso_code
4. Add 'owid_' prefix to distinguish from API data
5. Return aligned datasets
```

**Example:**
```python
# Before (OWID time series)
country       date        total_cases
USA           2020-01-01  1
USA           2020-01-02  5
...
USA           2024-08-14  103,000,000

# After (aligned)
country       owid_date   owid_total_cases
USA           2024-08-14  103,000,000
```

---

### 3.2 Country Mapping

**Function:** `create_country_mapping_table()`

**Purpose:** Show which countries can be merged

**Process:**
```python
1. Get unique countries from OWID
2. Get unique countries from API
3. Find union of all countries
4. For each country, check:
   - in_owid: Is it in OWID data?
   - in_api: Is it in API data?
   - can_merge: Is it in both?
5. Create mapping DataFrame
```

**Output:**
```python
country                in_owid  in_api  can_merge
Afghanistan            True     True    True
Albania                True     True    True
...
Taiwan                 True     False   False  # Only in OWID
Kosovo                 False    True    False  # Only in API
...
```

**Statistics:**
- Total countries: ~280
- In both datasets: 194
- Match rate: ~80%

---

### 3.3 Merge Datasets

**Function:** `merge_datasets()`

**Strategy:** Inner join on standardized country name

**Process:**
```python
1. Align temporal data
   → Get latest OWID data per country

2. Create country mapping
   → Identify mergeable countries

3. Inner join on 'country_standardized'
   → Only keep countries in both datasets

4. Combine columns:
   - OWID columns (owid_date, owid_total_cases, etc.)
   - API columns (api_current_cases, api_current_deaths, etc.)
   - Common columns (country_standardized, population, iso_code)

5. Calculate data gaps
   → Difference between OWID and API

6. Calculate derived metrics
   → Population-adjusted rates
   → Case fatality rates

7. Add quality flags
   → Large data gaps
   → Old data
   → Population anomalies
```

**Column Naming Convention:**
```python
# OWID columns
owid_date
owid_total_cases
owid_total_deaths
owid_population

# API columns
api_current_cases
api_current_deaths
api_tests
api_population

# Derived columns
current_cases_per_100k
current_deaths_per_100k
current_case_fatality_rate

# Gap columns
cases_data_gap
cases_data_gap_percent
deaths_data_gap
deaths_data_gap_percent
```

---

### 3.4 Calculate Data Gaps

**Purpose:** Quantify differences between OWID and API data

**Process:**
```python
1. Calculate absolute gap
   gap = api_current_cases - owid_total_cases

2. Calculate percentage gap
   gap_percent = (gap / owid_total_cases) * 100

3. Handle division by zero
   Use np.where to avoid errors

4. Round to 2 decimal places
```

**Example:**
```python
Country   OWID Cases   API Cases   Gap      Gap %
USA       103,000,000  103,436,829 436,829  0.42%
India     45,000,000   45,123,456  123,456  0.27%
Brazil    38,000,000   38,500,000  500,000  1.32%
```

**Interpretation:**
- Small gap (<1%): Data sources agree well
- Medium gap (1-10%): Some divergence
- Large gap (>10%): Significant difference (flagged)

---

### 3.5 Calculate Trend Metrics

**Function:** `calculate_trend_metrics()`

**Purpose:** Calculate 30-day trends from OWID historical data

**Process:**
```python
1. For each country:
   a. Get data from last 30 days
   b. Calculate average new cases
   c. Calculate average new deaths
   d. Calculate percent change

2. Add trend columns:
   - trend_30d_avg_new_cases
   - trend_30d_avg_new_deaths
   - trend_30d_percent_change
```

**Calculation:**
```python
# Get last 30 days of data
last_30_days = df[df['date'] >= (latest_date - timedelta(days=30))]

# Calculate average
avg_new_cases = last_30_days['new_cases'].mean()

# Calculate percent change
first_week_avg = last_30_days.head(7)['new_cases'].mean()
last_week_avg = last_30_days.tail(7)['new_cases'].mean()
percent_change = ((last_week_avg - first_week_avg) / first_week_avg) * 100
```

---

### 3.6 Add Quality Flags

**Purpose:** Flag data quality issues for user awareness

**Flags:**

**1. Large Data Gap (>10%):**
```python
if abs(cases_data_gap_percent) > 10:
    flag = "large_data_gap"
```

**2. Old Data (>90 days):**
```python
days_old = (today - owid_date).days
if days_old > 90:
    flag = "old_data"
```

**3. Population Anomaly:**
```python
if not (1000 < population < 2e9):
    flag = "population_anomaly"
```

**Output:**
```python
{
    "large_data_gaps": ["Country A", "Country B"],
    "old_data": ["Country C"],
    "population_anomalies": ["Country D"]
}
```

---

### 3.7 Generate Integration Summary

**Function:** `generate_integration_summary()`

**Purpose:** Create comprehensive report of integration process

**Output:**
```python
{
    "total_countries_merged": 194,
    "countries_with_trends": 189,
    "integration_timestamp": "2024-10-24T10:00:00",

    "source_details": {
        "owid_countries": 250,
        "api_countries": 231,
        "matched_countries": 194
    },

    "data_quality_flags": {
        "large_data_gaps": 12,
        "old_data": 8,
        "population_anomalies": 2
    },

    "coverage_statistics": {
        "match_rate": 0.81,
        "avg_data_gap_percent": 2.3,
        "median_data_age_days": 45
    }
}
```

---

## Final Output Dataset

### Structure

**194 countries × 30+ columns**

**Column Categories:**

**1. Identification:**
- country_standardized
- country (original)
- iso_code

**2. OWID Historical:**
- owid_date
- owid_total_cases
- owid_total_deaths
- owid_population

**3. API Current:**
- api_current_cases
- api_current_deaths
- api_tests
- api_population

**4. Derived Metrics:**
- current_cases_per_100k
- current_deaths_per_100k
- current_case_fatality_rate

**5. Data Gaps:**
- cases_data_gap
- cases_data_gap_percent
- deaths_data_gap
- deaths_data_gap_percent

**6. Trends:**
- trend_30d_avg_new_cases
- trend_30d_avg_new_deaths
- trend_30d_percent_change

**7. Quality Flags:**
- has_large_gap
- has_old_data
- has_population_anomaly

---

## Usage Example

### Running the Pipeline

```python
from covid_integration.data_loader import load_owid_data, load_disease_sh_data
from covid_integration.data_cleaner import clean_owid_data, clean_api_data
from covid_integration.data_merger import merge_datasets

# Step 1: Load data
owid_df = load_owid_data()      # ~5-7 seconds
api_df = load_disease_sh_data()  # ~1-2 seconds

# Step 2: Clean data
owid_clean = clean_owid_data(owid_df)     # Already done in load
api_clean = clean_api_data(api_df)

# Step 3: Merge data
merged_df, summary = merge_datasets(owid_clean, api_clean)

# Result: 194 countries ready for analysis
print(f"Merged {len(merged_df)} countries")
print(f"Columns: {list(merged_df.columns)}")
```

### Total Processing Time

- Data loading: ~8-10 seconds
- Data cleaning: <1 second
- Data merging: <1 second
- **Total: ~10-12 seconds**

---

## Key Achievements

✅ **High Match Rate:** 194/250 countries merged (77.6%)
✅ **Data Quality:** Comprehensive validation and flagging
✅ **Transparency:** All gaps and issues documented
✅ **Reproducible:** Consistent processing pipeline
✅ **Fast:** Complete pipeline runs in ~12 seconds
✅ **Tested:** 21+ integration tests covering all steps

---

## Data Quality Considerations

### Known Limitations

1. **Historical Data Only**
   - OWID data ends August 2024
   - No real-time updates

2. **Country Matching**
   - ~20% of countries don't match between sources
   - Different naming conventions
   - Some territories excluded

3. **Data Gaps**
   - Reporting delays vary by country
   - Different update frequencies
   - Missing data for some metrics

4. **Population Data**
   - May differ slightly between sources
   - Some estimates vs. actual counts

### How We Handle These

- **Transparent Reporting:** All gaps and issues flagged
- **Quality Metrics:** Comprehensive validation
- **User Warnings:** Dashboard shows data limitations
- **Documentation:** Clear about what data represents

---

## Related Documentation

- [Technical Details](TECHNICAL.md) - System architecture
- [Development Guide](DEVELOPMENT.md) - Running the pipeline
- [README](../README.md) - Quick start guide
