"""
COVID-19 Data Integration Dashboard

Interactive Streamlit application for exploring the integrated COVID-19 dataset.
Provides dynamic filtering, visualization, and analysis capabilities for the
merged OWID historical and disease.sh current data.
"""

import logging
import os
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from covid_integration.data_cleaner import clean_all_data

# Import our modules (after path setup)
from covid_integration.data_loader import load_all_data
from covid_integration.data_merger import integrate_covid_data

# Configure page
st.set_page_config(
    page_title="COVID-19 Data Integration Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configure logging to suppress verbose output
logging.getLogger().setLevel(logging.WARNING)


@st.cache_data
def load_and_integrate_data():
    """Load and integrate data with caching for performance."""
    try:
        with st.spinner("Loading and integrating COVID-19 data..."):
            # Load raw data
            owid_raw, api_raw, _ = load_all_data()

            # Clean data
            owid_clean, api_clean, _ = clean_all_data(owid_raw, api_raw)

            # Integrate data
            merged_data, integration_summary = integrate_covid_data(owid_clean, api_clean)

        return merged_data, integration_summary, True
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        return None, None, False


def create_overview_metrics(df):
    """Create overview metrics for the dashboard."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Countries Integrated",
            value=f"{len(df):,}",
            help="Total number of countries successfully merged",
        )

    with col2:
        total_cases = df["api_current_cases"].sum()
        st.metric(
            label="Total Cases (Aug 2024)",
            value=f"{total_cases:,.0f}",
            help="Sum of cases across all countries as of August 2024",
        )

    with col3:
        if "api_current_deaths" in df.columns:
            total_deaths = df["api_current_deaths"].sum()
            st.metric(
                label="Total Deaths (Aug 2024)",
                value=f"{total_deaths:,.0f}",
                help="Sum of deaths across all countries as of August 2024",
            )

    with col4:
        if "owid_data_age_days" in df.columns:
            avg_data_age = df["owid_data_age_days"].mean()
            st.metric(
                label="OWID Data Age",
                value=f"{avg_data_age:.0f} days",
                help="Average age of OWID historical data (from August 2024)",
            )


def create_country_comparison_plot(df, selected_countries, metric):
    """Create interactive country comparison plot."""
    if not selected_countries:
        st.warning("Please select at least one country for comparison.")
        return

    # Filter data for selected countries
    filtered_df = df[df["country_standardized"].isin(selected_countries)].copy()

    if metric == "Aug 2024 Cases vs Population":
        fig = px.scatter(
            filtered_df,
            x="api_current_cases",
            y="current_cases_per_100k",
            hover_name="country_standardized",
            size="owid_population",
            color="cases_data_gap_percent",
            title="Cases (Aug 2024): Total vs Per 100K Population",
            labels={
                "api_current_cases": "Total Cases (Aug 2024)",
                "current_cases_per_100k": "Cases per 100K Population",
                "cases_data_gap_percent": "Data Gap (%)",
            },
            color_continuous_scale="RdYlBu_r",
        )
        fig.update_layout(height=500)

    elif metric == "Data Gap Analysis (Historical vs API)":
        fig = px.bar(
            filtered_df,
            x="country_standardized",
            y="cases_data_gap_percent",
            color="cases_data_gap_percent",
            title="Data Gap: API vs OWID Historical Cases",
            labels={
                "country_standardized": "Country",
                "cases_data_gap_percent": "Data Gap (%)",
            },
            color_continuous_scale="RdYlBu_r",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="No Gap")
        fig.update_layout(height=500, xaxis_tickangle=-45)

    elif metric == "Case Fatality Rate (Aug 2024)":
        if "current_case_fatality_rate" in filtered_df.columns:
            fig = px.bar(
                filtered_df,
                x="country_standardized",
                y="current_case_fatality_rate",
                color="current_case_fatality_rate",
                title="Case Fatality Rate by Country (August 2024)",
                labels={
                    "country_standardized": "Country",
                    "current_case_fatality_rate": "Case Fatality Rate (%)",
                },
                color_continuous_scale="Reds",
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
        else:
            st.warning("Case fatality rate data not available.")
            return

    elif metric == "Population vs Cases (Aug 2024)":
        fig = px.scatter(
            filtered_df,
            x="owid_population",
            y="api_current_cases",
            hover_name="country_standardized",
            size="current_cases_per_100k",
            title="Population vs Cases (August 2024)",
            labels={
                "owid_population": "Population",
                "api_current_cases": "Cases (Aug 2024)",
                "current_cases_per_100k": "Cases per 100K",
            },
            log_x=True,
        )
        fig.update_layout(height=500)

    else:
        st.error(f"Unknown metric: {metric}")
        return

    st.plotly_chart(fig, use_container_width=True)


def create_data_quality_dashboard(df, integration_summary):
    """Create data quality overview dashboard."""
    st.subheader("📊 Data Quality Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Data Completeness by Source**")
        completeness_data = integration_summary.get("data_completeness", {})

        if completeness_data:
            completeness_df = pd.DataFrame(
                [
                    {
                        "Source": key.replace("_", " ").title(),
                        "Completeness (%)": stats["completeness_percent"],
                        "Available": stats["available"],
                        "Missing": stats["missing"],
                    }
                    for key, stats in completeness_data.items()
                ]
            )

            fig = px.bar(
                completeness_df,
                x="Source",
                y="Completeness (%)",
                color="Completeness (%)",
                title="Data Completeness by Source",
                color_continuous_scale="Greens",
                text="Completeness (%)",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Show detailed table
            st.dataframe(completeness_df, use_container_width=True)

    with col2:
        st.write("**Data Age Distribution**")
        if "owid_data_age_days" in df.columns:
            fig = px.histogram(
                df,
                x="owid_data_age_days",
                nbins=20,
                title="Distribution of OWID Data Age",
                labels={"owid_data_age_days": "Data Age (Days)", "count": "Number of Countries"},
            )
            fig.add_vline(
                x=df["owid_data_age_days"].mean(),
                line_dash="dash",
                annotation_text=f"Mean: {df['owid_data_age_days'].mean():.0f} days",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


def create_top_countries_interactive(df, metric_choice, top_n):
    """Create interactive top countries visualization."""
    if metric_choice == "Total Cases (Aug 2024)":
        top_countries = df.nlargest(top_n, "api_current_cases")
        fig = px.bar(
            top_countries,
            x="api_current_cases",
            y="country_standardized",
            orientation="h",
            title=f"Top {top_n} Countries by Total Cases (August 2024)",
            labels={"api_current_cases": "Cases (Aug 2024)", "country_standardized": "Country"},
            color="api_current_cases",
            color_continuous_scale="Blues",
        )

    elif metric_choice == "Cases per 100K (Aug 2024)":
        if "current_cases_per_100k" in df.columns:
            top_countries = df.nlargest(top_n, "current_cases_per_100k")
            fig = px.bar(
                top_countries,
                x="current_cases_per_100k",
                y="country_standardized",
                orientation="h",
                title=f"Top {top_n} Countries by Cases per 100K Population (August 2024)",
                labels={
                    "current_cases_per_100k": "Cases per 100K (Aug 2024)",
                    "country_standardized": "Country",
                },
                color="current_cases_per_100k",
                color_continuous_scale="Reds",
            )
        else:
            st.error("Cases per 100K data not available")
            return

    elif metric_choice == "Total Deaths (Aug 2024)":
        if "api_current_deaths" in df.columns:
            top_countries = df.nlargest(top_n, "api_current_deaths")
            fig = px.bar(
                top_countries,
                x="api_current_deaths",
                y="country_standardized",
                orientation="h",
                title=f"Top {top_n} Countries by Total Deaths (August 2024)",
                labels={
                    "api_current_deaths": "Deaths (Aug 2024)",
                    "country_standardized": "Country",
                },
                color="api_current_deaths",
                color_continuous_scale="Oranges",
            )
        else:
            st.error("Deaths data not available")
            return

    fig.update_layout(height=max(400, top_n * 25))
    st.plotly_chart(fig, use_container_width=True)


def create_world_map_visualization(df):
    """Create interactive world map visualization."""
    st.subheader("🗺️ Global COVID-19 Map (August 2024 Data)")

    map_metric = st.selectbox(
        "Select metric for map visualization:",
        [
            "Cases (Aug 2024)",
            "Cases per 100K (Aug 2024)",
            "Deaths (Aug 2024)",
            "Case Fatality Rate (Aug 2024)",
        ],
        key="map_metric",
    )

    # Map metric to column name
    metric_mapping = {
        "Cases (Aug 2024)": "api_current_cases",
        "Cases per 100K (Aug 2024)": "current_cases_per_100k",
        "Deaths (Aug 2024)": "api_current_deaths",
        "Case Fatality Rate (Aug 2024)": "current_case_fatality_rate",
    }

    column_name = metric_mapping[map_metric]

    # Find the correct ISO code column (could have suffixes after merge)
    iso_column = None
    for col in df.columns:
        if "iso_code" in col and not col.endswith("_api_meta") and not col.endswith("_owid_meta"):
            iso_column = col
            break

    # Fallback to checking for specific patterns
    if iso_column is None:
        for potential_col in ["iso_code_owid_meta", "iso_code_api_meta", "iso_code"]:
            if potential_col in df.columns:
                iso_column = potential_col
                break

    if iso_column is None or column_name not in df.columns:
        st.error(
            f"Required columns not found. Available ISO columns: {[col for col in df.columns if 'iso' in col.lower()]}"
        )
        return

    # Create a clean dataframe for the map with valid ISO codes
    map_df = df[df[iso_column].notna() & (df[iso_column] != "")].copy()

    if len(map_df) == 0:
        st.error("No valid ISO codes found for mapping.")
        return

    try:
        fig = px.choropleth(
            map_df,
            locations=iso_column,
            color=column_name,
            hover_name="country_standardized",
            hover_data={
                "api_current_cases": ":,.0f",
                "current_cases_per_100k": ":.1f"
                if "current_cases_per_100k" in df.columns
                else None,
                iso_column: False,
            },
            color_continuous_scale="Reds",
            title=f"Global {map_metric}",
            labels={column_name: map_metric},
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # Show debug info if needed
        with st.expander("🔧 Debug Info (Map Data)"):
            st.write(f"**Using ISO column:** `{iso_column}`")
            st.write(f"**Countries with valid ISO codes:** {len(map_df)}")
            st.write(f"**Sample ISO codes:** {map_df[iso_column].head().tolist()}")

    except Exception as e:
        st.error(f"Error creating map: {str(e)}")
        st.write("**Available columns for debugging:**")
        st.write(list(df.columns))


def main():
    """Main dashboard application."""
    st.title("🦠 COVID-19 Data Integration Dashboard")

    # Prominent data disclaimer
    st.error(
        "⚠️ **IMPORTANT DATA LIMITATION:** This dashboard shows historical data through August 2024 only. The 'current' metrics reflect the most recent available data from that time period, NOT real-time conditions."
    )

    st.markdown(
        """
    **Interactive exploration of integrated COVID-19 data from multiple sources**

    This dashboard combines historical data from Our World in Data (OWID) with API data from disease.sh,
    providing comprehensive insights into global COVID-19 patterns and data quality **as of August 2024**.
    """
    )

    # Load data
    merged_df, integration_summary, success = load_and_integrate_data()

    if not success or merged_df is None:
        st.error("Failed to load data. Please check your internet connection and try again.")
        return

    # Add data timestamp info after successful load
    if "owid_date" in merged_df.columns:
        latest_date = merged_df["owid_date"].max()
        st.info(
            f"📅 **Data Coverage:** Historical trends through {latest_date.strftime('%B %d, %Y')} | API snapshot from {latest_date.strftime('%B %Y')}"
        )

    # Sidebar filters
    st.sidebar.header("🔍 Filters & Options")

    # Add sidebar disclaimer too
    st.sidebar.error("⚠️ Data is historical through August 2024 only")

    # Country selection
    all_countries = sorted(merged_df["country_standardized"].unique())
    default_countries = ["United States", "India", "Brazil", "Germany", "France"]
    available_defaults = [c for c in default_countries if c in all_countries]

    selected_countries = st.sidebar.multiselect(
        "Select Countries for Comparison:",
        options=all_countries,
        default=available_defaults[:5],  # Limit to 5 for performance
        help="Choose countries to compare in the analysis charts",
    )

    # Metric filters
    case_threshold = st.sidebar.slider(
        "Minimum Cases (Aug 2024):",
        min_value=0,
        max_value=int(merged_df["api_current_cases"].max()),
        value=0,
        step=1000,
        help="Filter countries by minimum number of cases as of August 2024",
    )

    # Filter dataframe based on selections
    filtered_df = merged_df[merged_df["api_current_cases"] >= case_threshold].copy()

    # Overview metrics
    st.header("📈 Overview (August 2024 Data)")
    create_overview_metrics(filtered_df)

    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🏆 Top Countries",
            "📊 Country Comparison",
            "🗺️ World Map",
            "🔍 Data Quality",
            "📋 Raw Data",
            "📖 Data Definitions",
        ]
    )

    with tab1:
        st.subheader("🏆 Top Countries Analysis (August 2024)")

        col1, col2 = st.columns([2, 1])
        with col1:
            metric_choice = st.selectbox(
                "Select Metric:",
                ["Total Cases (Aug 2024)", "Cases per 100K (Aug 2024)", "Total Deaths (Aug 2024)"],
                key="top_countries_metric",
            )
        with col2:
            top_n = st.slider("Number of countries:", 5, 25, 15, key="top_n")

        create_top_countries_interactive(filtered_df, metric_choice, top_n)

    with tab2:
        st.subheader("📊 Country Comparison (August 2024 Data)")

        if selected_countries:
            comparison_metric = st.selectbox(
                "Select Analysis Type:",
                [
                    "Aug 2024 Cases vs Population",
                    "Data Gap Analysis (Historical vs API)",
                    "Case Fatality Rate (Aug 2024)",
                    "Population vs Cases (Aug 2024)",
                ],
                key="comparison_metric",
            )

            create_country_comparison_plot(filtered_df, selected_countries, comparison_metric)

            # Show summary table for selected countries
            if selected_countries:
                st.subheader("📋 Selected Countries Summary (August 2024)")
                summary_cols = [
                    "country_standardized",
                    "api_current_cases",
                    "current_cases_per_100k",
                    "owid_population",
                    "cases_data_gap_percent",
                ]
                available_cols = [col for col in summary_cols if col in filtered_df.columns]

                summary_df = filtered_df[
                    filtered_df["country_standardized"].isin(selected_countries)
                ][available_cols].round(2)

                st.dataframe(summary_df, use_container_width=True)
        else:
            st.info("👈 Please select countries in the sidebar to see comparisons.")

    with tab3:
        create_world_map_visualization(filtered_df)

    with tab4:
        create_data_quality_dashboard(filtered_df, integration_summary)

        # Show integration summary
        st.subheader("🔗 Integration Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.json(
                {
                    "Total Countries Merged": integration_summary.get("total_countries_merged", 0),
                    "Countries with Trends": integration_summary.get("countries_with_trends", 0),
                    "Integration Timestamp": integration_summary.get(
                        "integration_timestamp", "Unknown"
                    ),
                }
            )

        with col2:
            quality_flags = integration_summary.get("data_quality_flags", {})
            if quality_flags:
                st.json(quality_flags)

    with tab5:
        st.subheader("📋 Raw Integrated Dataset (August 2024)")
        st.markdown(
            f"**Dataset Shape:** {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns"
        )

        # Search functionality
        search_term = st.text_input("🔍 Search countries:", placeholder="Type country name...")

        if search_term:
            search_filtered = filtered_df[
                filtered_df["country_standardized"].str.contains(search_term, case=False, na=False)
            ]
            st.dataframe(search_filtered, use_container_width=True)
        else:
            # Show top 20 by default for performance
            st.dataframe(filtered_df.head(20), use_container_width=True)

            if len(filtered_df) > 20:
                st.info(
                    f"Showing first 20 rows of {len(filtered_df)} total. Use search to find specific countries."
                )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset as CSV",
            data=csv,
            file_name=f"covid_historical_data_aug2024_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

    with tab6:
        st.subheader("📖 Data Definitions & Methodology")

        st.markdown(
            """
        ### 🔍 Key Metrics Explained

        **OWID Data Age**
        Shows how many days have passed since the OWID historical data was last updated (August 14, 2024) to today's date.
        - **Example:** If today is June 21, 2025, the data age is ~329 days (about 11 months old)
        - **Why it matters:** Helps you understand how "stale" the historical data is
        - **Calculation:** `Today's Date - August 14, 2024`

        **Cases (Aug 2024)**
        Total cumulative COVID-19 cases reported by each country as of August 2024.
        - **Source:** disease.sh API snapshot from August 2024
        - **Note:** This is NOT current data - it's historical data from August 2024

        **Cases per 100K Population**
        Population-adjusted case rate to enable fair comparison between countries of different sizes.
        - **Formula:** `(Total Cases ÷ Population) × 100,000`
        - **Example:** If a country has 1,000 cases and 1 million people: `(1,000 ÷ 1,000,000) × 100,000 = 100 cases per 100K`

        **Data Gap (%)**
        Shows the percentage difference between API data and OWID historical data for the same time period.
        - **Positive gap:** API shows higher numbers than OWID (expected for newer data)
        - **Negative gap:** API shows lower numbers than OWID (potential data quality issue)
        - **Formula:** `((API Cases - OWID Cases) ÷ OWID Cases) × 100`

        **Case Fatality Rate (CFR)**
        Percentage of confirmed cases that resulted in death.
        - **Formula:** `(Total Deaths ÷ Total Cases) × 100`
        - **Important:** CFR varies by testing capacity, demographics, and healthcare quality
        - **Note:** This is crude CFR, not age-adjusted or outcome-adjusted
        """
        )

        st.markdown(
            """
        ### 📊 Data Sources & Integration

        **Our World in Data (OWID)**
        - **Coverage:** Historical time series through August 14, 2024
        - **Strengths:** Comprehensive, validated, time series data
        - **Limitations:** Not updated in real-time, stops at August 2024
        - **URL:** [ourworldindata.org/coronavirus](https://ourworldindata.org/coronavirus)

        **disease.sh API**
        - **Coverage:** Country-level snapshot data from August 2024
        - **Strengths:** Standardized format, population data included
        - **Limitations:** Point-in-time snapshot, no historical trends
        - **URL:** [disease.sh](https://disease.sh/)

        **Integration Methodology**
        1. **Country Standardization:** Applied 25+ country name mappings to align naming conventions
        2. **Temporal Alignment:** Used latest available OWID data per country (around August 14, 2024)
        3. **Data Validation:** Filtered out non-country aggregates (World, Europe, etc.)
        4. **Gap Analysis:** Calculated differences between sources for quality assessment
        5. **Derived Metrics:** Computed population-adjusted rates and quality indicators
        """
        )

        st.markdown(
            """
        ### ⚠️ Important Limitations & Caveats

        **Temporal Limitations**
        - **All data is historical through August 2024** - nothing is current/real-time
        - **Data age varies by country** - some countries may have stopped reporting earlier
        - **No trends beyond August 2024** - cannot show what happened after that date

        **Data Quality Considerations**
        - **Reporting differences:** Countries use different case definitions and testing strategies
        - **Underreporting:** Actual cases likely higher due to limited testing and asymptomatic cases
        - **Revised data:** Historical numbers may have been revised after our snapshot date
        - **Missing data:** Some countries have incomplete reporting for certain metrics

        **Comparative Analysis Limitations**
        - **Testing capacity varies:** Countries with more testing appear to have higher case rates
        - **Demographics matter:** CFR influenced by age structure and underlying health conditions
        - **Healthcare capacity:** Death rates affected by healthcare system quality and capacity
        - **Socioeconomic factors:** Reporting quality correlates with development level

        **Statistical Notes**
        - **Population data:** Based on latest available estimates, may not reflect 2024 populations exactly
        - **Rates and ratios:** Calculated using available data; missing values excluded from calculations
        - **Country matching:** 194 of ~276 total countries successfully matched between sources
        """
        )

        st.markdown(
            """
        ### 🎯 How to Use This Dashboard Effectively

        **For Comparative Analysis:**
        - Use "Cases per 100K" rather than total cases when comparing countries of different sizes
        - Consider data gaps when interpreting differences between countries
        - Focus on countries with high data completeness for most reliable comparisons

        **For Data Quality Assessment:**
        - Check the "Data Quality" tab before drawing conclusions
        - Review data gaps to understand source reliability
        - Consider data age when interpreting patterns

        **For Download & Further Analysis:**
        - Use the "Raw Data" tab to download the complete integrated dataset
        - Include data definitions and limitations in any reports using this data
        - Always cite the August 2024 limitation in presentations
        """
        )

        st.info(
            """
        **💡 Professional Tip:** This dashboard demonstrates data integration methodology and quality assessment practices.
        The comprehensive metadata and limitation documentation shown here represents best practices for
        data science projects in consulting and research environments.
        """
        )

    # Footer
    st.markdown("---")
    st.markdown(
        """
    **⚠️ HISTORICAL DATA ONLY - NOT CURRENT CONDITIONS**

    **Data Sources & Timeframe:**
    - Historical: [Our World in Data](https://ourworldindata.org/coronavirus) (through August 14, 2024)
    - API Snapshot: [disease.sh API](https://disease.sh/) (August 2024 snapshot)

    **Dashboard Generated:** {timestamp}
    """.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M UTC")
        )
    )


if __name__ == "__main__":
    main()
