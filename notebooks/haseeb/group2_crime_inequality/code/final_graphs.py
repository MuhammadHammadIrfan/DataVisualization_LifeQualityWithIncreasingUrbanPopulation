import os
import sys
from typing import List

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import seaborn as sns

# Allow running this script directly by adding the project root to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from notebooks.haseeb.group2_crime_inequality.code.basic_trends_correlations import (  # type: ignore  # noqa: E501
    load_and_merge_data,
)
from notebooks.haseeb.group2_crime_inequality.code.region_correlations import (  # type: ignore  # noqa: E501
    add_region_info,
)


def ensure_output_dir(base_path: str = ".") -> str:
    """Ensure the shared images output directory exists and return its path."""

    images_dir = os.path.join(base_path, "group2_crime_inequality", "images")
    os.makedirs(images_dir, exist_ok=True)
    return images_dir


def subdir(images_dir: str, name: str) -> str:
    """Create and return a named subdirectory under the main images dir."""

    path = os.path.join(images_dir, name)
    os.makedirs(path, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# 1) Inequalityurbanization interaction heatmap (violent crime)
# ---------------------------------------------------------------------------


def add_tertiles(df: pd.DataFrame) -> pd.DataFrame:
    """Add inequality and urbanization tertile labels used for heatmaps."""

    out = df.copy()
    out = out.dropna(subset=["gini_coef", "urban_pop_perc"])

    out["gini_tertile"] = pd.qcut(
        out["gini_coef"], 3, labels=["Low inequality", "Mid inequality", "High inequality"]
    )
    out["urban_tertile"] = pd.qcut(
        out["urban_pop_perc"], 3, labels=["Low urban", "Mid urban", "High urban"]
    )

    return out


def plot_inequality_urban_interaction_heatmap(
    df: pd.DataFrame, images_dir: str, target_col: str, fname: str
) -> None:
    """Heatmap of average violence metric by inequality and urbanization tertiles."""

    df_tertiles = add_tertiles(df)

    pivot = df_tertiles.groupby(["gini_tertile", "urban_tertile"])[target_col].mean().unstack()

    # Ensure consistent ordering of labels on axes
    row_order: List[str] = ["Low inequality", "Mid inequality", "High inequality"]
    col_order: List[str] = ["Low urban", "Mid urban", "High urban"]
    pivot = pivot.reindex(index=row_order, columns=col_order)

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="Reds",
        cbar_kws={"label": f"Average {target_col.replace('_', ' ')}"},
    )
    plt.xlabel("Urbanization level")
    plt.ylabel("Inequality level")
    plt.title(f"{target_col.replace('_', ' ').title()} by inequalityurbanization buckets")
    plt.tight_layout()
    inter_dir = subdir(images_dir, "interactions")
    out_path = os.path.join(inter_dir, fname)
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# 2) Perceived crime vs homicide (core scatter)
# ---------------------------------------------------------------------------


def plot_perceptions_vs_actual(df_region: pd.DataFrame, images_dir: str) -> None:
    """Perceptions of criminality vs actual homicide, coloured by region."""

    core_dir = subdir(images_dir, "core")

    plt.figure(figsize=(6.5, 4.5))
    ax1 = plt.gca()
    sns.scatterplot(
        data=df_region,
        x="perceptions_crime",
        y="homicide_rate",
        hue="region",
        alpha=0.7,
        ax=ax1,
    )
    ax1.set_xlabel("Perceptions of criminality (worse \u2192 higher)")
    ax1.set_ylabel("Homicide rate (index)")
    ax1.set_title("Perceived crime vs homicide")

    plt.tight_layout()
    out_path = os.path.join(core_dir, "perceptions_vs_actual_violence.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# 3) Police presence vs societal safety (by region)
# ---------------------------------------------------------------------------


def plot_police_vs_safety_by_region(df_region: pd.DataFrame, images_dir: str) -> None:
    """Police capacity vs GPI safety, coloured by region."""

    core_dir = subdir(images_dir, "core")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=df_region,
        x="police_rate",
        y="gpi_safety",
        hue="region",
        alpha=0.7,
    )
    plt.xlabel("Police capacity index (higher = more officers per capita)")
    plt.ylabel("GPI safety & security (higher = less safe)")
    plt.title("Police presence vs societal safety (by region)")
    plt.tight_layout()
    out_path = os.path.join(core_dir, "police_vs_safety_by_region.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# 4) Inequality vs homicide, coloured by urbanization
# ---------------------------------------------------------------------------


def plot_scatter_gini_vs_homicide(df: pd.DataFrame, images_dir: str) -> None:
    """Replicates the final inequality vs homicide scatter used in the report."""

    sns.set(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="gini_coef",
        y="homicide_rate",
        hue="urban_pop_perc",
        palette="magma",
        alpha=0.6,
    )
    plt.xlabel("Gini coefficient (2021 prices)")
    plt.ylabel("Homicide rate")
    plt.title("Inequality vs Homicide Rate (colored by urbanization)")
    plt.tight_layout()

    out_path = os.path.join(images_dir, "scatter_gini_vs_homicide.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# 5) Key countries risk plane: violence pressure vs state response
# ---------------------------------------------------------------------------


def plot_key_country_risk_plane(df: pd.DataFrame, images_dir: str) -> None:
    """Presentation-ready chart comparing influential countries on a risk plane."""

    # Build the influential-country set based on urban population exposed to violence
    summary = (
        df.groupby("country")
        .agg(
            pop_mean=("total_pop", "mean"),
            urban_mean=("urban_pop_perc", "mean"),
            violent_mean=("violent_crime", "mean"),
        )
        .dropna()
    )

    if summary.empty:
        return

    summary["urban_pop"] = summary["pop_mean"] * (summary["urban_mean"] / 100.0)
    violent_threshold = summary["violent_mean"].median()
    summary = summary[summary["violent_mean"] >= violent_threshold]
    summary["score"] = summary["urban_pop"] * summary["violent_mean"]

    # Fixed, presentation-oriented selection including the United States,
    # constrained to countries that actually exist in the merged data.
    preferred_order = [
        "Brazil",
        "Colombia",
        "Russian Federation",
        "Turkiye",
        "United States",
        "China",
        "Indonesia",
        "Germany",
    ]
    available_countries = set(df["country"].unique().tolist())
    key_countries = [c for c in preferred_order if c in available_countries]

    if not key_countries:
        return

    df_key = df[df["country"].isin(key_countries)].copy()

    # Global 0\u20131 scaling for inputs to the composite indices (GPI-style columns only)
    def scaled(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series([0.5] * len(s), index=s.index)
        return (s - mn) / (mx - mn)

    df_key["perceptions_s"] = scaled(df_key["perceptions_crime"])
    df_key["violent_demos_s"] = scaled(df_key["violent_demonstrations"])
    df_key["homicide_s"] = scaled(df_key["homicide_rate"])
    df_key["violent_s"] = scaled(df_key["violent_crime"])
    df_key["arms_s"] = scaled(df_key["access_small_arms"])
    df_key["police_s"] = scaled(df_key["police_rate"])
    df_key["incarceration_s"] = scaled(df_key["incarceration_rate"])

    # Composite indices
    df_key["violence_pressure"] = df_key[
        ["homicide_s", "violent_s", "violent_demos_s", "perceptions_s", "arms_s"]
    ].mean(axis=1)
    df_key["state_force"] = df_key[["police_s", "incarceration_s"]].mean(axis=1)

    # Latest year per country (usually 2020)
    latest_per_country = (
        df_key.sort_values("year")
        .groupby("country")
        .tail(1)
        .set_index("country")
    )

    # Colour scale: GPI safety (higher = less safe) -> redder
    safety_vals = latest_per_country["gpi_safety"].astype(float)
    norm = Normalize(vmin=safety_vals.min(), vmax=safety_vals.max())
    cmap = plt.cm.get_cmap("RdYlGn_r")

    # Bubble size: more violence pressure = bigger bubble
    vp = latest_per_country["violence_pressure"]
    vp_min, vp_max = vp.min(), vp.max()
    if vp_max == vp_min:
        bubble_sizes = pd.Series([200.0] * len(vp), index=vp.index)
    else:
        bubble_sizes = 100.0 + 400.0 * (vp - vp_min) / (vp_max - vp_min)

    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    # Background trajectories over time in grey
    for country in key_countries:
        sub = df_key[df_key["country"] == country].sort_values("year")
        ax.plot(
            sub["violence_pressure"],
            sub["state_force"],
            color="lightgrey",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )

    # Scatter of latest positions
    sc = ax.scatter(
        latest_per_country["violence_pressure"],
        latest_per_country["state_force"],
        s=bubble_sizes,
        c=cmap(norm(safety_vals)),
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )

    # Country labels next to latest bubbles
    for country, row in latest_per_country.iterrows():
        ax.text(
            row["violence_pressure"] + 0.01,
            row["state_force"] + 0.01,
            country,
            fontsize=8,
            fontweight="bold",
        )

    # Median lines to split into four conceptual quadrants
    x_med = latest_per_country["violence_pressure"].median()
    y_med = latest_per_country["state_force"].median()
    ax.axvline(x_med, color="grey", linestyle="--", linewidth=0.8)
    ax.axhline(y_med, color="grey", linestyle="--", linewidth=0.8)

    ax.set_xlabel(
        "Violence pressure index (scaled 0\u20131)\n"
        "(homicide, violent crime, violent demonstrations, perceptions of criminality,\n"
        " access to small arms)"
    )
    ax.set_ylabel(
        "State force index (scaled 0\u20131)\n"
        "(police rate and incarceration rate)"
    )
    ax.set_title(
        "Influential countries: violence pressure vs state response\n"
        "Latest year positions (colour = societal safety, size = violence pressure)"
    )

    # Colour bar for GPI safety scale
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.02,
    )
    cbar.set_label("GPI safety & security (higher = less safe)")

    plt.tight_layout()
    countries_dir = subdir(images_dir, "countries")
    out_path = os.path.join(
        countries_dir, "key_countries_risk_plane_inequality_violence.png"
    )
    plt.savefig(out_path, dpi=300)
    plt.close()


# ---------------------------------------------------------------------------
# Main entry point: build the merged dataset and generate only the 5 finals.
# ---------------------------------------------------------------------------


def main() -> None:
    base_path = "."
    images_dir = ensure_output_dir(base_path)

    df = load_and_merge_data(base_path)
    df_region = add_region_info(df, base_path)

    print(f"Merged rows (all): {len(df)}")
    print(f"Rows with region info: {df_region['region'].notna().sum()}")

    # 1) Heatmap: inequality x urbanization -> violent crime
    plot_inequality_urban_interaction_heatmap(
        df,
        images_dir,
        target_col="violent_crime",
        fname="heatmap_ineq_urban_violentcrime.png",
    )

    # 2) Perceived crime vs homicide (core scatter)
    plot_perceptions_vs_actual(df_region, images_dir)

    # 3) Police presence vs societal safety, by region
    plot_police_vs_safety_by_region(df_region, images_dir)

    # 4) Inequality vs homicide, coloured by urbanization
    plot_scatter_gini_vs_homicide(df, images_dir)

    # 5) Key countries risk plane: violence pressure vs state response
    plot_key_country_risk_plane(df, images_dir)

    print(f"\nFinal selected plots saved to: {images_dir}")


if __name__ == "__main__":
    main()
