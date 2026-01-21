import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH = "winemag-data-130k-v2.csv"
JSON_PATH = "winemag-data-130k-v2.json"
OUTPUT_DIR = "plots"
MIN_COUNTRY_SAMPLE_SIZE = 50  # minimum wines per country to be included


# ---------------------------------------------------------------------------
# Data loading and basic validation
# ---------------------------------------------------------------------------

def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load CSV and JSON versions of the wine dataset.
    """
    df_csv = pd.read_csv(CSV_PATH)
    # JSON is line-delimited
    df_json = pd.read_json(JSON_PATH, lines=True)
    return df_csv, df_json


def validate_datasets(df_csv: pd.DataFrame, df_json: pd.DataFrame) -> None:
    """
    Perform lightweight consistency checks between CSV and JSON versions.
    Prints summary information; does not raise unless files are clearly incompatible.
    """
    print("=== Dataset Shapes ===")
    print(f"CSV shape : {df_csv.shape}")
    print(f"JSON shape: {df_json.shape}")

    common_cols = sorted(set(df_csv.columns) & set(df_json.columns))
    print("\n=== Common Columns Between CSV and JSON ===")
    print(common_cols)

    key_cols = [c for c in ["country", "price", "points", "province", "region_1", "variety", "winery"] if c in common_cols]
    print("\nKey columns used for basic comparison:")
    print(key_cols)

    # Compare basic price distributions as a sanity check
    if "price" in df_csv.columns and "price" in df_json.columns:
        print("\n=== Price Distribution (CSV) ===")
        print(df_csv["price"].describe())
        print("\n=== Price Distribution (JSON) ===")
        print(df_json["price"].describe())


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Coerce price to numeric
    - Drop rows with missing or non-positive price
    - Drop rows with missing country
    """
    df = df.copy()

    # Ensure price is numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    initial_rows = len(df)

    # Drop invalid prices and missing countries
    df = df[
        df["price"].notna()
        & (df["price"] > 0)
        & df["country"].notna()
    ].copy()

    print(f"\nRows before cleaning: {initial_rows:,}")
    print(f"Rows after  cleaning: {len(df):,}")

    return df


# ---------------------------------------------------------------------------
# Country-level aggregation
# ---------------------------------------------------------------------------

def compute_country_stats(df: pd.DataFrame, min_sample: int = MIN_COUNTRY_SAMPLE_SIZE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Group by country and compute:
    - Mean price
    - Median price
    - Price standard deviation
    - Wine count per country
    - Mean points (where available)
    """
    has_points = "points" in df.columns

    agg_dict = {
        "price": ["mean", "median", "std", "count"],
    }
    if has_points:
        agg_dict["points"] = ["mean"]

    grouped = df.groupby("country").agg(agg_dict)

    # Flatten column index
    new_cols = []
    for col, stat in grouped.columns:
        if col == "price" and stat == "mean":
            new_cols.append("mean_price")
        elif col == "price" and stat == "median":
            new_cols.append("median_price")
        elif col == "price" and stat == "std":
            new_cols.append("std_price")
        elif col == "price" and stat == "count":
            new_cols.append("n_wines")
        elif col == "points" and stat == "mean":
            new_cols.append("mean_points")
        else:
            new_cols.append(f"{col}_{stat}")

    grouped.columns = new_cols
    grouped = grouped.reset_index()

    filtered = grouped[grouped["n_wines"] >= min_sample].copy()

    print(f"\nNumber of countries (all): {len(grouped)}")
    print(f"Number of countries (n_wines >= {min_sample}): {len(filtered)}")

    return grouped, filtered


# ---------------------------------------------------------------------------
# Best value calculation (high points, low price)
# ---------------------------------------------------------------------------

def compute_best_value_countries(df: pd.DataFrame, min_sample: int = MIN_COUNTRY_SAMPLE_SIZE) -> pd.DataFrame:
    """
    Compute a simple value metric: points per unit price.
    High points and low price -> large value score.
    """
    if "points" not in df.columns:
        print("\nNo 'points' column in data; skipping best-value calculation.")
        return pd.DataFrame()

    df_val = df.copy()
    df_val = df_val[df_val["points"].notna()].copy()

    # Value metric: points per dollar
    df_val["value_score"] = df_val["points"] / df_val["price"]

    value_by_country = (
        df_val.groupby("country")["value_score"]
        .mean()
        .reset_index()
    )

    counts = df_val.groupby("country")["price"].count().reset_index(name="n_wines")
    value_by_country = value_by_country.merge(counts, on="country", how="left")
    value_by_country = value_by_country[value_by_country["n_wines"] >= min_sample].copy()

    value_by_country = value_by_country.sort_values("value_score", ascending=False)

    print("\n=== Top 10 Best-Value Countries (points per price) ===")
    print(value_by_country.head(10).to_string(index=False))

    return value_by_country


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_boxplot_top10_by_count(df: pd.DataFrame) -> None:
    """
    Boxplot of price by country for top 10 countries by wine count.
    """
    ensure_output_dir()

    counts = (
        df["country"]
        .value_counts()
        .head(10)
        .index
    )
    top10_df = df[df["country"].isin(counts)].copy()

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=top10_df,
        x="country",
        y="price"
    )
    plt.xticks(rotation=45, ha="right")
    plt.title("Wine Price Distribution by Country (Top 10 by Count)")
    plt.ylabel("Price (USD)")
    plt.tight_layout()
    plt.yscale("log")  # optional log-scale to handle outliers

    out_path = os.path.join(OUTPUT_DIR, "boxplot_price_by_country_top10.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved boxplot to {out_path}")


def plot_bar_avg_price_per_country(country_stats: pd.DataFrame) -> None:
    """
    Bar chart of average price per country (filtered by min sample size).
    """
    ensure_output_dir()

    df_bar = country_stats.sort_values("mean_price", ascending=False)

    plt.figure(figsize=(14, 7))
    sns.barplot(
        data=df_bar,
        x="country",
        y="mean_price",
        color="steelblue"
    )
    plt.xticks(rotation=90)
    plt.ylabel("Average Price (USD)")
    plt.title("Average Wine Price by Country")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "bar_mean_price_by_country.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved bar chart to {out_path}")


def plot_scatter_price_vs_points(df: pd.DataFrame) -> None:
    """
    Scatter plot of price vs points, colored by country.
    To keep this readable, limit to top 6 countries by count.
    """
    if "points" not in df.columns:
        print("\nNo 'points' column in data; skipping scatter plot.")
        return

    ensure_output_dir()

    top_countries = df["country"].value_counts().head(6).index
    sub = df[df["country"].isin(top_countries)].copy()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=sub,
        x="points",
        y="price",
        hue="country",
        alpha=0.5
    )
    plt.yscale("log")  # log scale for price to handle outliers
    plt.ylabel("Price (USD, log scale)")
    plt.title("Price vs Points by Country (Top 6 by Count)")
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "scatter_price_vs_points_by_country.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved scatter plot to {out_path}")


# ---------------------------------------------------------------------------
# Textual conclusions
# ---------------------------------------------------------------------------

def print_country_rankings(country_stats: pd.DataFrame) -> None:
    """
    Print rankings for:
    - Top / bottom countries by average price
    - Top countries by median price
    """
    by_mean_desc = country_stats.sort_values("mean_price", ascending=False)
    by_median_desc = country_stats.sort_values("median_price", ascending=False)

    print("\n=== Top 15 Countries by Mean Price (USD) ===")
    print(by_mean_desc.head(15).to_string(index=False))

    print("\n=== Bottom 15 Countries by Mean Price (USD) ===")
    print(by_mean_desc.tail(15).to_string(index=False))

    print("\n=== Top 15 Countries by Median Price (USD) ===")
    print(by_median_desc.head(15).to_string(index=False))


def print_focus_comparisons(country_stats: pd.DataFrame, value_stats: pd.DataFrame) -> None:
    """
    Explicitly compare:
    - Italy vs South Africa
    - France, USA, Spain, Chile, Argentina
    """
    focus_countries = ["Italy", "South Africa", "France", "US", "Spain", "Chile", "Argentina"]
    focus = country_stats[country_stats["country"].isin(focus_countries)].copy()

    print("\n=== Focus Countries (Mean/Median Price, Count) ===")
    print(
        focus[["country", "mean_price", "median_price", "std_price", "n_wines"]]
        .sort_values("mean_price", ascending=False)
        .to_string(index=False)
    )

    if not value_stats.empty:
        focus_val = value_stats[value_stats["country"].isin(focus_countries)].copy()
        if not focus_val.empty:
            print("\n=== Focus Countries by Best-Value Score (points per price) ===")
            print(
                focus_val[["country", "value_score", "n_wines"]]
                .sort_values("value_score", ascending=False)
                .to_string(index=False)
            )


def print_overall_conclusions(country_stats: pd.DataFrame, value_stats: pd.DataFrame) -> None:
    """
    Print concise conclusions:
    - Is South Africa expensive or affordable?
    - Is Italy overpriced or good value?
    - Which country dominates high price points?
    - Which country offers the best wine experience based on price?
    """
    by_mean_desc = country_stats.sort_values("mean_price", ascending=False).reset_index(drop=True)
    by_median_desc = country_stats.sort_values("median_price", ascending=False).reset_index(drop=True)

    # Most and least expensive (by mean price)
    most_expensive = by_mean_desc.iloc[0]
    cheapest = by_mean_desc.iloc[-1]

    # Helper to get stats for a given country
    def get_country_row(name: str):
        rows = country_stats[country_stats["country"] == name]
        return rows.iloc[0] if not rows.empty else None

    italy = get_country_row("Italy")
    sa = get_country_row("South Africa")

    # Overall averages (of country means)
    overall_mean_of_means = country_stats["mean_price"].mean()

    print("\n=== CONCLUSIONS ===")

    # South Africa: expensive or affordable?
    if sa is not None:
        sa_rank = (by_mean_desc["country"] == "South Africa").idxmax() + 1
        total_countries = len(by_mean_desc)
        sa_position = "expensive" if sa["mean_price"] > overall_mean_of_means else "affordable"
        print(
            f"- South Africa: mean price ${sa['mean_price']:.2f}, "
            f"ranked {sa_rank} out of {total_countries} countries -> relatively {sa_position}."
        )
    else:
        print("- South Africa: not enough data after filtering to evaluate.")

    # Italy: overpriced or good value?
    if italy is not None:
        italy_rank = (by_mean_desc["country"] == "Italy").idxmax() + 1
        italy_position = "expensive" if italy["mean_price"] > overall_mean_of_means else "affordable"
        conclusion = f"- Italy: mean price ${italy['mean_price']:.2f}, ranked {italy_rank} -> relatively {italy_position}"

        if not value_stats.empty and "Italy" in set(value_stats["country"]):
            italy_value_rank = (
                value_stats.sort_values("value_score", ascending=False)["country"] == "Italy"
            ).idxmax() + 1
            conclusion += f"; value rank (points/price) #{italy_value_rank} among countries."

        print(conclusion + ".")
    else:
        print("- Italy: not enough data after filtering to evaluate.")

    # Which country dominates high price points?
    print(
        f"- Most expensive country (by average price): {most_expensive['country']} "
        f"with mean price ${most_expensive['mean_price']:.2f}."
    )

    # Best wine experience by price (best value)
    if not value_stats.empty:
        best_value = value_stats.iloc[0]
        print(
            f"- Best-value country (highest points per price): {best_value['country']} "
            f"(value score {best_value['value_score']:.3f})."
        )
    else:
        print("- Best-value ranking could not be computed (no points information).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load and validate
    df_csv, df_json = load_datasets()
    validate_datasets(df_csv, df_json)

    # Use CSV as primary source for analysis
    df = clean_data(df_csv)

    # Core stats
    _, country_stats_filtered = compute_country_stats(df, min_sample=MIN_COUNTRY_SAMPLE_SIZE)

    # Best value
    value_stats = compute_best_value_countries(df, min_sample=MIN_COUNTRY_SAMPLE_SIZE)

    # Plots
    plot_boxplot_top10_by_count(df)
    plot_bar_avg_price_per_country(country_stats_filtered)
    plot_scatter_price_vs_points(df)

    # Text summaries / tables
    print_country_rankings(country_stats_filtered)
    print_focus_comparisons(country_stats_filtered, value_stats)
    print_overall_conclusions(country_stats_filtered, value_stats)


if __name__ == "__main__":
    main()

