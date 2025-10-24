# COVID-19 Data Integration Dashboard

> **Interactive dashboard integrating multiple COVID-19 data sources with quality assessment and visualizations.**

⚠️ **Note:** This project uses historical data through August 2024 only.

## Quick Start

### Run Locally

```bash
# 1. Install dependencies
poetry install

# 2. Launch the dashboard
poetry run streamlit run streamlit_app.py
```

Dashboard will be available at **http://localhost:8501**

### Requirements

- Python 3.9+
- Poetry ([install](https://python-poetry.org/docs/#installation))

## What's Inside

- **📊 Interactive Dashboard** - Streamlit app with country comparisons and visualizations
- **🔗 Data Integration** - Combines OWID historical data with disease.sh API
- **✅ Quality Checks** - Data validation, country name harmonization, gap analysis
- **🧪 Tests** - 21+ integration tests covering data pipeline

## Project Structure

```
covid-data-challenge/
├── streamlit_app.py           # Main dashboard app
├── src/covid_integration/     # Data processing modules
│   ├── data_loader.py         # Load data from sources
│   ├── data_cleaner.py        # Standardize and validate
│   ├── data_merger.py         # Integrate datasets
│   └── visualizer.py          # Create visualizations
├── tests/                     # Test suite
└── docs/                      # Additional documentation
```

## Common Tasks

### Run Tests
```bash
poetry run pytest tests/ -v
```

### Run Individual Components
```bash
# Test data loading
poetry run python src/covid_integration/data_loader.py

# Generate static visualizations
poetry run python src/covid_integration/visualizer.py
```

### Code Quality
```bash
poetry run black src/ tests/ streamlit_app.py
poetry run flake8 src/ tests/ streamlit_app.py
```

## Data Sources

- **[Our World in Data](https://ourworldindata.org/coronavirus)** - Historical COVID-19 data
- **[disease.sh API](https://disease.sh/)** - Current country-level statistics

## Documentation

- [Technical Details](docs/TECHNICAL.md) - Architecture, data pipeline, methodology
- [Development Guide](docs/DEVELOPMENT.md) - Code quality, testing, contributing
- [Future Roadmap](docs/ROADMAP.md) - Planned enhancements and next steps

## Contact

**Lina L. Faller, Ph.D.**
Email: lina.faller@gmail.com

---

*Production-ready data engineering demonstration emphasizing data quality and methodological transparency.*
