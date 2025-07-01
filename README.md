# COVID-19 Data Integration Challenge

**A comprehensive data engineering and visualization project demonstrating production-ready data integration practices.**

## 🎯 Project Overview

This project showcases end-to-end data engineering capabilities through the integration of multiple COVID-19 data sources, creating a unified dataset with comprehensive quality assessment and interactive visualizations. Built as a technical demonstration of modern data science and engineering practices.

### 🌟 Key Challenges Demonstrated

- **Multi-source Data Integration**: Combining CSV historical data (Our World in Data) with real-time JSON API data (disease.sh)
- **Data Quality & Standardization**: Harmonizing country names, handling missing data, and temporal alignment
- **Statistical Analysis**: Trend calculations, gap analysis, and population-adjusted metrics
- **Production-Ready Engineering**: Comprehensive testing, professional code standards, and modular architecture
- **Interactive Data Applications**: Full-stack Streamlit dashboard with publication-quality visualizations
- **Professional Documentation**: Complete methodology transparency and data limitation disclosure

### 📊 Technical Stack

- **Language**: Python 3.9+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Testing**: pytest
- **Code Quality**: black, isort, flake8
- **Dependency Management**: Poetry
- **Version Control**: Git with conventional commits

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Poetry (for dependency management)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd covid-data-challenge

# Install dependencies with Poetry
poetry install

# Activate the virtual environment
poetry shell
```

## 🧪 Running Tests

```bash
# Run the complete test suite
poetry run pytest tests/test_integration.py -v

# Run tests with coverage
poetry run pytest tests/test_integration.py -v --cov=src

# Run specific test categories
poetry run pytest tests/test_integration.py::TestDataLoader -v
poetry run pytest tests/test_integration.py::TestDataCleaner -v
poetry run pytest tests/test_integration.py::TestDataMerger -v
```

## 📊 Running the Interactive Dashboard

```bash
# Launch the Streamlit dashboard
poetry run streamlit run streamlit_app.py

# Launch with debug mode (shows diagnostic information)
poetry run streamlit run streamlit_app.py -- --debug

# Or set environment variable for debug mode
STREAMLIT_DEBUG=true poetry run streamlit run streamlit_app.py

# Dashboard will be available at http://localhost:8501
```

### Dashboard Features

- **Top Countries Analysis**: Ranked visualizations with population-adjusted metrics
- **Country Comparison**: Interactive filtering and multi-metric analysis
- **Global Map Visualization**: Choropleth maps with metric selection
- **Data Quality Assessment**: Integration statistics and completeness analysis
- **Raw Data Explorer**: Searchable dataset with export functionality
- **Data Definitions**: Comprehensive methodology and limitation documentation
- **Debug Mode**: Diagnostic information for data pipeline troubleshooting (when enabled)

## 🛠️ Running Individual Components

### Data Integration Pipeline

```bash
# Test data loading
poetry run python src/covid_integration/data_loader.py

# Test data cleaning
poetry run python src/covid_integration/data_cleaner.py

# Test data merging
poetry run python src/covid_integration/data_merger.py

# Generate static visualizations
poetry run python src/covid_integration/visualizer.py
```

### Code Quality Checks

```bash
# Format code with Black
poetry run black src/ tests/ streamlit_app.py

# Organize imports with isort
poetry run isort src/ tests/ streamlit_app.py

# Run linting with flake8
poetry run flake8 src/ tests/ streamlit_app.py --max-line-length=100
```

## 📁 Project Structure

```
covid-data-challenge/
├── .flake8                        # Linting configuration
├── outputs/                       # Generated visualizations and datasets
├── poetry.lock
├── pyproject.toml                 # Poetry configuration and dependencies
├── README.md
├── src
│   ├── __init__.py
│   └── covid_integration
│       ├── __init__.py
│       ├── config
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   └── logging_config.py
│       ├── data_cleaner.py        # Data standardization and quality assessment
│       ├── data_loader.py         # Multi-source data loading (CSV + JSON API)
│       ├── data_merger.py         # Temporal alignment and integration logic
│       └── visualizer.py          # Publication-quality static visualizations
├── streamlit_app.py               # Interactive dashboard application
└── tests
    ├── __init__.py
    ├── conftest.py                # Test configuration
    └── test_integration.py        # Comprehensive test suite (21+ tests)
```

## 📈 Data Sources & Methodology

### Data Sources

- **Historical Data**: [Our World in Data](https://ourworldindata.org/coronavirus) - Comprehensive time series through August 2024
- **Current Data**: [disease.sh API](https://disease.sh/) - Country-level snapshot data

### Integration Process

1. **Data Loading**: Automated retrieval from CSV files and REST APIs
2. **Standardization**: Country name harmonization using 25+ mapping rules
3. **Quality Assessment**: Missing data analysis and completeness scoring
4. **Temporal Alignment**: Latest data extraction with gap analysis
5. **Derived Metrics**: Population-adjusted rates and trend calculations
6. **Validation**: Comprehensive data quality reporting

### Key Achievements

- **194 countries successfully integrated** (80%+ match rate)
- **Comprehensive data quality transparency**
- **Professional-grade documentation and limitations disclosure**
- **Production-ready error handling and logging**

## ⚠️ Important Data Limitations

**This project uses historical data through August 2024 only.** All "current" references in the codebase refer to August 2024 data, not real-time conditions. This limitation is clearly documented throughout the application interface.

## 🔄 Next Steps & Future Enhancements

### Infrastructure & DevOps
- **Docker Containerization**: Create multi-stage Dockerfile for consistent deployment
- **GitHub Actions CI/CD**: Automated testing, linting, and deployment pipeline
- **Pre-commit Hooks**: Automatic code formatting and quality checks
- **Container Registry**: Docker image publishing for easy deployment
- **Version Management**: Semantic versioning with automated tagging and release notes

### Technical Enhancements
- **Real-time Data Pipeline**: Integration with live COVID data APIs
- **Advanced Analytics**: Time series forecasting and anomaly detection
- **Performance Optimization**: Database backend for large-scale data processing
- **API Development**: REST API endpoints for programmatic data access

### Visualization & UX
- **Advanced Dashboards**: Multi-dimensional analysis with drill-down capabilities
- **Export Formats**: PDF reports and PowerBI/Tableau integration
- **Mobile Optimization**: Responsive design for mobile devices
- **User Authentication**: Role-based access and personalized dashboards

### Data Science Extensions
- **Machine Learning**: Predictive modeling for case trends
- **Geospatial Analysis**: Advanced mapping with demographic overlays
- **Statistical Modeling**: Correlation analysis and causal inference
- **Data Lake Architecture**: Scalable storage for multiple data sources

### Release Management
- **Semantic Versioning**: Following semver (e.g., v1.0.0, v1.1.0, v2.0.0) for clear version tracking
- **Automated Releases**: GitHub Actions for automated version bumping and release creation
- **Changelog Generation**: Automated release notes from conventional commits
- **Version Pinning**: Reproducible builds with locked dependency versions

## 🧹 Code Quality Standards

This project follows professional development practices:

- **Test Coverage**: 21+ comprehensive tests covering all major functionality
- **Code Formatting**: Black and isort for consistent style
- **Linting**: flake8 for code quality enforcement
- **Type Safety**: Strategic use of type hints for critical functions
- **Documentation**: Comprehensive docstrings and inline comments
- **Error Handling**: Graceful failure modes with informative logging

## 🤝 Development Acknowledgments

This project was developed with assistance from Claude (Anthropic) for code review, architectural guidance, and best practices validation. All technical decisions, problem-solving approaches, and implementation strategies reflect my own data engineering experience and judgment.

## 📄 License

This project is intended as a technical demonstration and coding challenge submission.

## 🤝 Contact

**Lina L. Faller, Ph.D.**  
Email: lina.faller@gmail.com

---

*This project demonstrates production-ready data engineering practices suitable for consulting environments, emphasizing data quality, methodological transparency, and user-focused design.*