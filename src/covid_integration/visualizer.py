"""
COVID-19 Data Visualization Module

This module creates publication-quality static visualizations from the integrated
COVID-19 dataset. Includes trend analysis, comparative plots, and data quality
visualizations that tell compelling data stories.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import centralized constants
try:
    # Relative import (when used as module)
    from .config.constants import (
        COLORS,
        DEFAULT_DPI,
        DEFAULT_FIGURE_SIZE,
    )
except ImportError:
    # Absolute import (when run directly)
    from covid_integration.config.constants import (
        COLORS,
        DEFAULT_DPI,
        DEFAULT_FIGURE_SIZE,
    )

# Configure logging
logger = logging.getLogger(__name__)

# Set style for publication-quality plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_plot_style():
    """Set up consistent styling for all plots."""
    plt.rcParams.update(
        {
            "figure.figsize": DEFAULT_FIGURE_SIZE,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "savefig.dpi": DEFAULT_DPI,
            "savefig.bbox": "tight",
        }
    )


def create_top_countries_plot(merged_df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """
    Create a horizontal bar chart of top countries by current cases.

    Args:
        merged_df: Integrated COVID dataset
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    setup_plot_style()

    # Get top 15 countries by current cases
    top_countries = merged_df.nlargest(15, "api_current_cases")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

    # Plot 1: Total cases
    y_pos = np.arange(len(top_countries))
    bars1 = ax1.barh(
        y_pos, top_countries["api_current_cases"] / 1e6, color=COLORS["primary"], alpha=0.8
    )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_countries["country_standardized"])
    ax1.set_xlabel("Current Cases (Millions)")
    ax1.set_title("Top 15 Countries by Total COVID-19 Cases", fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(
            width + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}M",
            ha="left",
            va="center",
            fontsize=9,
        )

    # Plot 2: Cases per 100k population
    if "current_cases_per_100k" in merged_df.columns:
        bars2 = ax2.barh(
            y_pos, top_countries["current_cases_per_100k"], color=COLORS["secondary"], alpha=0.8
        )

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(top_countries["country_standardized"])
        ax2.set_xlabel("Cases per 100K Population")
        ax2.set_title("Cases per 100K Population (Same Countries)", fontweight="bold")
        ax2.grid(axis="x", alpha=0.3)

        # Add value labels on bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(
                width + 100,
                bar.get_y() + bar.get_height() / 2,
                f"{width:,.0f}",
                ha="left",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "top_countries_cases.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created top countries plot: {output_path}")
    return str(output_path)


def create_data_gap_analysis_plot(merged_df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """
    Create visualization showing data gaps between OWID and API sources.

    Args:
        merged_df: Integrated COVID dataset
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    setup_plot_style()

    # Filter to countries with meaningful data gaps
    gap_data = merged_df[
        (merged_df["cases_data_gap_percent"].abs() > 1)
        & (merged_df["cases_data_gap_percent"].notna())
    ].copy()

    if len(gap_data) == 0:
        logger.warning("No significant data gaps found for visualization")
        return None

    # Sort by absolute gap percentage and take top 20
    gap_data["abs_gap_percent"] = gap_data["cases_data_gap_percent"].abs()
    gap_data = gap_data.nlargest(20, "abs_gap_percent")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Plot 1: Data gap percentages
    colors = [
        COLORS["success"] if x > 0 else COLORS["warning"]
        for x in gap_data["cases_data_gap_percent"]
    ]
    bars = ax1.barh(
        range(len(gap_data)), gap_data["cases_data_gap_percent"], color=colors, alpha=0.7
    )

    ax1.set_yticks(range(len(gap_data)))
    ax1.set_yticklabels(gap_data["country_standardized"])
    ax1.set_xlabel("Data Gap (%)")
    ax1.set_title("Data Gaps: API Current vs OWID Historical Cases", fontweight="bold")
    ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax1.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label_x = width + (1 if width > 0 else -1)
        ax1.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{width:+.1f}%",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=8,
        )

    # Plot 2: OWID data age vs gap size
    if "owid_data_age_days" in merged_df.columns:
        scatter_data = merged_df[
            merged_df["owid_data_age_days"].notna() & merged_df["cases_data_gap_percent"].notna()
        ]

        scatter = ax2.scatter(
            scatter_data["owid_data_age_days"],
            scatter_data["cases_data_gap_percent"].abs(),
            c=scatter_data["api_current_cases"],
            s=60,
            alpha=0.6,
            cmap="viridis",
        )

        ax2.set_xlabel("OWID Data Age (Days)")
        ax2.set_ylabel("Absolute Data Gap (%)")
        ax2.set_title("Data Gap vs Data Age (Color = Current Cases)", fontweight="bold")
        ax2.grid(alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Current Cases", rotation=270, labelpad=20)

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "data_gap_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created data gap analysis plot: {output_path}")
    return str(output_path)


def create_trend_analysis_plot(merged_df: pd.DataFrame, output_dir: str = "outputs") -> str:
    """
    Create visualization of 30-day trends for selected countries.

    Args:
        merged_df: Integrated COVID dataset
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    setup_plot_style()

    # Filter to countries with trend data
    trend_data = merged_df[
        (merged_df["avg_daily_new_cases"].notna()) & (merged_df["cases_change_percent"].notna())
    ].copy()

    if len(trend_data) == 0:
        logger.warning("No trend data available for visualization")
        return None

    # Get the trend period for labeling
    trend_end_date = None
    trend_start_date = None
    if "trend_end_date" in merged_df.columns and "trend_start_date" in merged_df.columns:
        trend_end_date = merged_df["trend_end_date"].max()
        trend_start_date = merged_df["trend_start_date"].min()

    # Create date range string for subtitle
    if trend_end_date is not None and trend_start_date is not None:
        date_range = f"Data from {trend_start_date.strftime('%b %d')} - {trend_end_date.strftime('%b %d, %Y')}"
    else:
        date_range = "Historical 30-day period ending Aug 2024"

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Add main title with time period
    fig.suptitle(f"COVID-19 Trend Analysis\n{date_range}", fontsize=16, fontweight="bold", y=0.98)

    # Plot 1: Cases percentage change distribution
    ax1.hist(
        trend_data["cases_change_percent"],
        bins=30,
        color=COLORS["primary"],
        alpha=0.7,
        edgecolor="black",
    )
    ax1.set_xlabel("30-Day Cases Change (%)")
    ax1.set_ylabel("Number of Countries")
    ax1.set_title("Distribution of 30-Day Case Changes\n(Historical Period)", fontweight="bold")
    ax1.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No Change")

    # Calculate and add interpretation
    increasing_countries = (trend_data["cases_change_percent"] > 0).sum()
    decreasing_countries = (trend_data["cases_change_percent"] < 0).sum()
    median_change = trend_data["cases_change_percent"].median()

    interpretation1 = f"Increasing: {increasing_countries} countries\nDecreasing: {decreasing_countries} countries\nMedian change: {median_change:+.1f}%"
    ax1.text(
        0.02,
        0.98,
        interpretation1,
        transform=ax1.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        fontsize=9,
    )
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Deaths percentage change distribution
    deaths_data = trend_data[trend_data["deaths_change_percent"].notna()]
    if len(deaths_data) > 0:
        ax2.hist(
            deaths_data["deaths_change_percent"],
            bins=30,
            color=COLORS["secondary"],
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("30-Day Deaths Change (%)")
        ax2.set_ylabel("Number of Countries")
        ax2.set_title(
            "Distribution of 30-Day Death Changes\n(Historical Period)", fontweight="bold"
        )
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No Change")

        # Calculate and add interpretation for deaths
        deaths_increasing = (deaths_data["deaths_change_percent"] > 0).sum()
        deaths_decreasing = (deaths_data["deaths_change_percent"] < 0).sum()
        deaths_median = deaths_data["deaths_change_percent"].median()

        interpretation2 = f"Increasing: {deaths_increasing} countries\nDecreasing: {deaths_decreasing} countries\nMedian change: {deaths_median:+.1f}%"
        ax2.text(
            0.02,
            0.98,
            interpretation2,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
            fontsize=9,
        )
        ax2.legend()
        ax2.grid(alpha=0.3)

    # Plot 3: Average daily new cases (top 15 countries)
    top_daily_cases = trend_data.nlargest(15, "avg_daily_new_cases")
    bars3 = ax3.bar(
        range(len(top_daily_cases)),
        top_daily_cases["avg_daily_new_cases"],
        color=COLORS["accent"],
        alpha=0.8,
    )
    ax3.set_xticks(range(len(top_daily_cases)))
    ax3.set_xticklabels(top_daily_cases["country_standardized"], rotation=45, ha="right")
    ax3.set_ylabel("Average Daily New Cases")
    ax3.set_title(
        "Top 15 Countries by Avg Daily New Cases\n(30-day Historical Period)", fontweight="bold"
    )
    ax3.grid(axis="y", alpha=0.3)

    # Add interpretation for daily cases
    max_daily = top_daily_cases["avg_daily_new_cases"].iloc[0]
    top_country = top_daily_cases["country_standardized"].iloc[0]
    total_avg = trend_data["avg_daily_new_cases"].mean()

    interpretation3 = (
        f"Leader: {top_country}\n({max_daily:.0f} daily avg)\nGlobal avg: {total_avg:.1f} cases/day"
    )
    ax3.text(
        0.02,
        0.98,
        interpretation3,
        transform=ax3.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        fontsize=9,
    )

    # Add value labels on bars
    for bar in bars3:
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + height * 0.01,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 4: Cases vs Deaths change correlation
    valid_changes = trend_data[
        (trend_data["cases_change_percent"].notna()) & (trend_data["deaths_change_percent"].notna())
    ]

    if len(valid_changes) > 0:
        scatter = ax4.scatter(
            valid_changes["cases_change_percent"],
            valid_changes["deaths_change_percent"],
            c=valid_changes["api_current_cases"],
            s=60,
            alpha=0.6,
            cmap="plasma",
        )

        ax4.set_xlabel("Cases Change (%)")
        ax4.set_ylabel("Deaths Change (%)")
        ax4.set_title(
            "Cases vs Deaths Change Correlation\n(Historical 30-day Period)", fontweight="bold"
        )
        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax4.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax4.grid(alpha=0.3)

        # Calculate correlation and add interpretation
        correlation = valid_changes["cases_change_percent"].corr(
            valid_changes["deaths_change_percent"]
        )

        # Determine quadrants
        q1_countries = (
            (valid_changes["cases_change_percent"] > 0)
            & (valid_changes["deaths_change_percent"] > 0)
        ).sum()  # Both increasing
        q2_countries = (
            (valid_changes["cases_change_percent"] < 0)
            & (valid_changes["deaths_change_percent"] > 0)
        ).sum()  # Cases down, deaths up
        q3_countries = (
            (valid_changes["cases_change_percent"] < 0)
            & (valid_changes["deaths_change_percent"] < 0)
        ).sum()  # Both decreasing
        q4_countries = (
            (valid_changes["cases_change_percent"] > 0)
            & (valid_changes["deaths_change_percent"] < 0)
        ).sum()  # Cases up, deaths down

        # Calculate data availability
        total_countries = len(merged_df)
        countries_with_data = len(valid_changes)
        missing_data_count = total_countries - countries_with_data

        interpretation4 = f"Correlation: {correlation:.3f}\nData available: {countries_with_data}/{total_countries}\nBoth ↑: {q1_countries}  Both ↓: {q3_countries}\nMixed: {q2_countries + q4_countries}  Missing: {missing_data_count}"
        ax4.text(
            0.02,
            0.98,
            interpretation4,
            transform=ax4.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
            fontsize=9,
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label("Current Total Cases", rotation=270, labelpad=20)

    # Add disclaimer text at bottom
    fig.text(
        0.5,
        0.02,
        "Note: Trend analysis based on historical OWID data. Not indicative of current conditions.",
        ha="center",
        fontsize=10,
        style="italic",
        color="gray",
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)  # Make room for title and disclaimer

    # Save plot
    output_path = Path(output_dir) / "trend_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created trend analysis plot: {output_path}")
    return str(output_path)


def create_data_quality_summary_plot(
    merged_df: pd.DataFrame, integration_summary: Dict, output_dir: str = "outputs"
) -> str:
    """
    Create a comprehensive data quality summary visualization.

    Args:
        merged_df: Integrated COVID dataset
        integration_summary: Summary statistics from integration
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    setup_plot_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Data completeness by source
    completeness_data = integration_summary.get("data_completeness", {})
    if completeness_data:
        sources = []
        completeness_pcts = []

        for col, stats in completeness_data.items():
            source_name = col.replace("_", " ").title()
            sources.append(source_name)
            completeness_pcts.append(stats["completeness_percent"])

        bars1 = ax1.bar(
            sources,
            completeness_pcts,
            color=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["warning"]][
                : len(sources)
            ],
            alpha=0.8,
        )

        ax1.set_ylabel("Completeness (%)")
        ax1.set_title("Data Completeness by Source", fontweight="bold")
        ax1.set_ylim(0, 105)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, pct in zip(bars1, completeness_pcts):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        # Rotate x-axis labels if needed
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")

    # Plot 2: Data age distribution
    if "owid_data_age_days" in merged_df.columns:
        age_data = merged_df["owid_data_age_days"].dropna()
        ax2.hist(age_data, bins=20, color=COLORS["info"], alpha=0.7, edgecolor="black")
        ax2.set_xlabel("OWID Data Age (Days)")
        ax2.set_ylabel("Number of Countries")
        ax2.set_title("Distribution of Data Age", fontweight="bold")
        ax2.axvline(
            x=age_data.mean(),
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {age_data.mean():.0f} days",
        )
        ax2.legend()
        ax2.grid(alpha=0.3)

    # Plot 3: Case fatality rates by region (if available)
    if "current_case_fatality_rate" in merged_df.columns:
        cfr_data = merged_df[merged_df["current_case_fatality_rate"].notna()]

        # Create CFR distribution
        ax3.hist(
            cfr_data["current_case_fatality_rate"],
            bins=25,
            color=COLORS["success"],
            alpha=0.7,
            edgecolor="black",
        )
        ax3.set_xlabel("Case Fatality Rate (%)")
        ax3.set_ylabel("Number of Countries")
        ax3.set_title("Distribution of Case Fatality Rates", fontweight="bold")
        ax3.axvline(
            x=cfr_data["current_case_fatality_rate"].median(),
            color="blue",
            linestyle="--",
            linewidth=2,
            label=f'Median: {cfr_data["current_case_fatality_rate"].median():.2f}%',
        )
        ax3.legend()
        ax3.grid(alpha=0.3)

    # Plot 4: Integration summary stats
    summary_stats = {
        "Countries Merged": integration_summary.get("total_countries_merged", 0),
        "With Trend Data": integration_summary.get("countries_with_trends", 0),
        "Large Case Gaps": integration_summary.get("data_quality_flags", {}).get(
            "countries_with_large_case_gaps", 0
        ),
        "Old OWID Data": integration_summary.get("data_quality_flags", {}).get(
            "countries_with_old_owid_data", 0
        ),
    }

    stat_names = list(summary_stats.keys())
    stat_values = list(summary_stats.values())

    bars4 = ax4.bar(
        stat_names,
        stat_values,
        color=[COLORS["primary"], COLORS["accent"], COLORS["warning"], COLORS["success"]],
        alpha=0.8,
    )

    ax4.set_ylabel("Count")
    ax4.set_title("Integration Summary Statistics", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, val in zip(bars4, stat_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(stat_values) * 0.01,
            f"{val}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    # Save plot
    output_path = Path(output_dir) / "data_quality_summary.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Created data quality summary plot: {output_path}")
    return str(output_path)


def generate_all_visualizations(
    merged_df: pd.DataFrame, integration_summary: Dict, output_dir: str = "outputs"
) -> Dict[str, str]:
    """
    Generate all static visualizations from the integrated dataset.

    Args:
        merged_df: Integrated COVID dataset
        integration_summary: Summary statistics from integration
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping plot names to file paths
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(exist_ok=True)

    logger.info("Generating comprehensive visualization suite...")

    plot_paths = {}

    try:
        # Generate each plot
        plot_paths["top_countries"] = create_top_countries_plot(merged_df, output_dir)
        plot_paths["data_gap_analysis"] = create_data_gap_analysis_plot(merged_df, output_dir)
        plot_paths["data_quality_summary"] = create_data_quality_summary_plot(
            merged_df, integration_summary, output_dir
        )

        # Filter out any None values (plots that couldn't be created)
        plot_paths = {k: v for k, v in plot_paths.items() if v is not None}

        logger.info(f"Successfully generated {len(plot_paths)} visualizations")

    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")
        raise

    return plot_paths


if __name__ == "__main__":
    # Test the visualization functions
    from data_cleaner import clean_all_data
    from data_loader import load_all_data
    from data_merger import integrate_covid_data

    try:
        # Load, clean, and integrate data
        logger.info("Loading and processing data for visualization...")
        owid_raw, api_raw, _ = load_all_data()
        owid_clean, api_clean, _ = clean_all_data(owid_raw, api_raw)
        merged_data, integration_summary = integrate_covid_data(owid_clean, api_clean)

        # Generate visualizations
        plot_paths = generate_all_visualizations(merged_data, integration_summary)

        print("\n=== Visualization Generation Complete ===")
        print(f"Generated {len(plot_paths)} plots:")
        for plot_name, path in plot_paths.items():
            print(f"  {plot_name}: {path}")

        print("\nDataset summary:")
        print(f"  Countries: {len(merged_data)}")
        print(f"  Columns: {len(merged_data.columns)}")
        print(f"  Integration summary available: {bool(integration_summary)}")

    except Exception as e:
        logger.error(f"Visualization test failed: {e}")
        raise
