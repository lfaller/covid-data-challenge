# Development Guide

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Poetry ([installation guide](https://python-poetry.org/docs/#installation))
- Git

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd covid-data-challenge

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Running the Project

### Interactive Dashboard

```bash
# Launch the Streamlit app
poetry run streamlit run streamlit_app.py

# With debug mode (shows diagnostic info)
poetry run streamlit run streamlit_app.py -- --debug

# Or use environment variable
STREAMLIT_DEBUG=true poetry run streamlit run streamlit_app.py
```

Dashboard available at: http://localhost:8501

**Dashboard Features:**
- Top Countries Analysis
- Country Comparison (with country selector in tab)
- World Map Visualization
- Data Quality Dashboard
- Raw Data Explorer
- Data Definitions

### Individual Components

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

## Testing

### Run All Tests

```bash
# Run full test suite
poetry run pytest tests/test_integration.py -v

# With coverage report
poetry run pytest tests/test_integration.py -v --cov=src

# Show coverage in terminal
poetry run pytest tests/test_integration.py --cov=src --cov-report=term-missing
```

### Run Specific Tests

```bash
# Test data loader only
poetry run pytest tests/test_integration.py::TestDataLoader -v

# Test data cleaner only
poetry run pytest tests/test_integration.py::TestDataCleaner -v

# Test data merger only
poetry run pytest tests/test_integration.py::TestDataMerger -v

# Single specific test
poetry run pytest tests/test_integration.py::TestDataLoader::test_load_disease_sh_data_success -v
```

### Test Coverage

Current coverage: **21+ integration tests** covering:
- ✅ Data loading (OWID, disease.sh API)
- ✅ Country name standardization
- ✅ Data filtering and validation
- ✅ Data merging and integration
- ✅ Metric calculations
- ✅ Error handling

## Code Quality

### Formatting

```bash
# Format all code with Black
poetry run black src/ tests/ streamlit_app.py

# Check formatting without changing files
poetry run black --check src/ tests/ streamlit_app.py

# Format specific file
poetry run black src/covid_integration/data_loader.py
```

### Import Sorting

```bash
# Sort imports with isort
poetry run isort src/ tests/ streamlit_app.py

# Check without changing
poetry run isort --check-only src/ tests/ streamlit_app.py
```

### Linting

```bash
# Run flake8 linter
poetry run flake8 src/ tests/ streamlit_app.py --max-line-length=100

# Check specific file
poetry run flake8 src/covid_integration/data_loader.py
```

### All Quality Checks

```bash
# Run all quality checks at once
poetry run black src/ tests/ streamlit_app.py && \
poetry run isort src/ tests/ streamlit_app.py && \
poetry run flake8 src/ tests/ streamlit_app.py --max-line-length=100
```

## Project Structure

```
covid-data-challenge/
├── .flake8                        # Linting configuration
├── .gitignore
├── README.md
├── pyproject.toml                 # Poetry dependencies
├── poetry.lock                    # Locked dependencies
│
├── docs/                          # Documentation
│   ├── TECHNICAL.md               # Technical architecture
│   ├── DEVELOPMENT.md             # This file
│   └── ROADMAP.md                 # Future enhancements
│
├── src/
│   └── covid_integration/
│       ├── __init__.py            # Package init with import utilities
│       ├── config/
│       │   ├── constants.py       # Configuration constants
│       │   └── logging_config.py  # Logging setup
│       ├── utils/                 # Utility functions
│       │   ├── __init__.py
│       │   └── data_filters.py    # Centralized filtering
│       ├── data_loader.py         # Load data from sources
│       ├── data_cleaner.py        # Clean and standardize
│       ├── data_merger.py         # Integrate datasets
│       └── visualizer.py          # Create visualizations
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Test configuration
│   └── test_integration.py        # Integration tests
│
├── streamlit_app.py               # Interactive dashboard
└── outputs/                       # Generated files (gitignored)
```

## Code Standards

### Naming Conventions

- **Functions**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Docstrings

Use Google-style docstrings:

```python
def calculate_metrics(df: pd.DataFrame, metric_type: str) -> pd.DataFrame:
    """
    Calculate population-adjusted metrics for countries.

    Args:
        df: DataFrame with country data
        metric_type: Type of metric to calculate ('cases' or 'deaths')

    Returns:
        DataFrame with calculated metrics added

    Raises:
        ValueError: If metric_type is not recognized
    """
    pass
```

### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, List, Optional

def process_data(
    data: pd.DataFrame,
    columns: List[str],
    options: Optional[Dict[str, any]] = None
) -> pd.DataFrame:
    pass
```

### Logging

Use centralized logging:

```python
from covid_integration import get_logger

logger = get_logger(__name__)

logger.info("Processing started")
logger.warning("Missing data found")
logger.error("Failed to process", exc_info=True)
```

## Contributing

### Workflow

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/bug-description
   # or
   git checkout -b refactor/code-improvement
   ```

2. **Make changes**
   - Write code
   - Add/update tests
   - Update documentation

3. **Quality checks**
   ```bash
   poetry run black src/ tests/ streamlit_app.py
   poetry run isort src/ tests/ streamlit_app.py
   poetry run flake8 src/ tests/ streamlit_app.py
   poetry run pytest tests/ -v
   ```

4. **Commit**
   ```bash
   git add .
   git commit -m "feat: Add new feature"
   # or
   git commit -m "fix: Fix bug in data loading"
   # or
   git commit -m "refactor: Improve code organization"
   ```

5. **Push and create PR**
   ```bash
   git push -u origin feature/my-feature
   gh pr create --title "Feature: My Feature" --body "Description..."
   ```

### Commit Message Format

Follow conventional commits:

- `feat:` - New feature
- `fix:` - Bug fix
- `refactor:` - Code refactoring
- `docs:` - Documentation changes
- `test:` - Test updates
- `chore:` - Maintenance tasks

### Pull Request Checklist

- [ ] Code follows project style (black, isort, flake8)
- [ ] Tests added/updated and passing
- [ ] Documentation updated if needed
- [ ] No breaking changes (or clearly documented)
- [ ] PR description explains changes
- [ ] Related issues linked

## Debugging

### Enable Debug Mode

```python
# In any module
import os
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

if DEBUG:
    print(f"Debug: {variable}")
```

### Streamlit Debug Mode

```bash
# Run with debug flag
poetry run streamlit run streamlit_app.py -- --debug

# Or use environment variable
STREAMLIT_DEBUG=true poetry run streamlit run streamlit_app.py
```

Shows:
- Available countries in dropdown
- Country name matching
- Data source information
- Integration statistics

### Logging Levels

```python
# In logging_config.py
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Set to DEBUG for verbose output
LOG_LEVEL=DEBUG poetry run python src/covid_integration/data_loader.py
```

## Common Issues

### Poetry Install Fails

```bash
# Clear cache
poetry cache clear pypi --all

# Re-install
poetry install
```

### Import Errors

```bash
# Reinstall package in development mode
poetry install

# Or activate shell
poetry shell
```

### Streamlit Won't Start

```bash
# Check port availability
lsof -i :8501

# Use different port
poetry run streamlit run streamlit_app.py --server.port 8502
```

### Tests Fail

```bash
# Update test dependencies
poetry update pytest

# Clear pytest cache
rm -rf .pytest_cache
poetry run pytest tests/ -v
```

## Useful Commands

```bash
# Show dependency tree
poetry show --tree

# Update a specific package
poetry update pandas

# Add new dependency
poetry add package-name

# Add dev dependency
poetry add --group dev package-name

# Remove dependency
poetry remove package-name

# Export requirements.txt
poetry export -f requirements.txt --output requirements.txt
```

## Resources

- [Poetry Documentation](https://python-poetry.org/docs/)
- [Pytest Documentation](https://docs.pytest.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Black Code Formatter](https://black.readthedocs.io/)
