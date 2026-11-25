"""Plot UHS injection utilisation and unused capacity by model instance.

The left panel reproduces the mean marginal full-load-hour curve. The right
panel retains the individual country--weather cases, making zero utilisation
visible even when it disappears from an averaged curve. A black ring indicates
that the selected withdrawal-to-injection ratio (WTIR) reaches its upper bound.

Example (from the repository root)::

    pixi run python -m scripts.results_analysis.plot_injection_capacity_utilisation `
        --per-case results/capacity_utilisation/storage_marginal_power_salt_cavern_NLD_GBR_DEU_per_case.csv `
        --design results/design_outcomes/storage_design_outcomes_NLD_DEU_GBR_points.csv `
        --output results/capacity_utilisation/injection_capacity_utilisation_SC.pdf

Only matplotlib, numpy and pandas are required. No NetCDF files need to be
opened; both panels use the previously exported results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Publication settings and editable defaults
# ---------------------------------------------------------------------------

METHOD_LABELS = {"B1": "B", "P1": "P", "F1": "F", "FX1": "FX"}
METHOD_ORDER = ("B1", "P1", "F1", "FX1")
COUNTRY_ORDER = ("NLD", "DEU", "GBR")
COLORMAP_POSITIONS = (0.08, 0.35, 0.60, 0.79)

FIGURE_WIDTH_MM = 190.0  # Elsevier double-column width
FIGURE_HEIGHT_MM = 72.0
FONT_SIZE_PT = 9.0
ZERO_TOLERANCE_HOURS = 1e-6
WTIR_BOUND = 10.0
WTIR_TOLERANCE = 1e-3


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing required columns: {', '.join(missing)}")


def selected_mask(series: pd.Series) -> pd.Series:
    """Accept boolean or string-valued selected flags from a CSV export."""
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(("true", "false", "1", "0")).all():
        raise ValueError("'selected' must contain only True or False values")
    return values.isin(("true", "1"))


def load_case_profiles(
    per_case_file: Path,
    technology: str,
    methods: list[str],
    countries: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(per_case_file)
    required = {
        "country", "experiment", "weather", "technology", "metric",
        "slice_index", "capacity_percentile", "value", "selected",
    }
    require_columns(df, required, per_case_file)

    df = df.loc[
        (df["technology"] == technology)
        & (df["metric"] == "injection_full_load_hours_per_year")
        & df["country"].isin(countries)
        & df["experiment"].isin(methods)
        & selected_mask(df["selected"]),
    ].copy()
    if df.empty:
        raise ValueError("No selected injection profiles match the requested filters")

    case_key = ["country", "experiment", "weather"]
    slice_key = case_key + ["slice_index"]
    if df.duplicated(slice_key).any():
        raise ValueError("Duplicate injection-capacity slices within a model case")

    for col in ("value", "capacity_percentile", "slice_index"):
        df[col] = pd.to_numeric(df[col], errors="raise")
    if df["value"].isna().any() or df["capacity_percentile"].isna().any():
        raise ValueError("Selected injection profiles contain missing numerical values")
    if (df["value"] < -ZERO_TOLERANCE_HOURS).any():
        raise ValueError("Selected injection profiles contain negative full-load hours")

    # The method averages like-for-like percentiles. Detect partial profiles,
    # which could otherwise change the effective sample between percentile bins.
    expected = None
    for case, group in df.groupby(case_key):
        percentiles = np.sort(group["capacity_percentile"].to_numpy())
        if expected is None:
            expected = percentiles
        elif not np.allclose(percentiles, expected, atol=1e-9, rtol=0):
            raise ValueError(f"Incompatible capacity-percentile grid for {case}")
    return df


def load_designs(
    design_file: Path,
    technology: str,
    methods: list[str],
    countries: list[str],
) -> pd.DataFrame:
    df = pd.read_csv(design_file)
    required = {"country", "experiment", "weather", "technology", "WTIR"}
    require_columns(df, required, design_file)
    df = df.loc[
        (df["technology"] == technology)
        & df["experiment"].isin(methods)
        & df["country"].isin(countries),
        ["country", "experiment", "weather", "WTIR"],
    ].copy()
    key = ["country", "experiment", "weather"]
    if df.duplicated(key).any():
        raise ValueError(
            "The design CSV contains multiple technology designs per case; "
            "aggregate or select them explicitly before plotting."
        )
    df["WTIR"] = pd.to_numeric(df["WTIR"], errors="raise")
    return df


def build_results(
    cases: pd.DataFrame,
    designs: pd.DataFrame,
    wtir_bound: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    key = ["country", "experiment", "weather"]
    grouped = cases.groupby(key, sort=False)
    unused = grouped.agg(
        unused_pct=(
            "value",
            lambda values: 100.0 * np.mean(
                np.abs(np.asarray(values, dtype=float)) <= ZERO_TOLERANCE_HOURS
            ),
        ),
        capacity_increments=("value", "size"),
    ).reset_index()
    unused = unused.merge(designs, on=key, how="left", validate="one_to_one")
    if unused["WTIR"].isna().any():
        missing = unused.loc[unused["WTIR"].isna(), key]
        raise ValueError(f"Missing WTIR design data for:\n{missing.to_string(index=False)}")
    if (unused["WTIR"] > wtir_bound + WTIR_TOLERANCE).any():
        raise ValueError("A selected WTIR exceeds the supplied upper WTIR bound")
    unused["at_wtir_bound"] = np.isclose(
        unused["WTIR"], wtir_bound, atol=WTIR_TOLERANCE, rtol=0.0
    )

    curves = cases.groupby(["experiment", "capacity_percentile"], as_index=False)[
        "value"
    ].mean()
    return curves, unused


def draw_figure(
    curves: pd.DataFrame,
    unused: pd.DataFrame,
    methods: list[str],
    countries: list[str],
    wtir_bound: float,
    output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE_PT,
            "axes.labelsize": FONT_SIZE_PT,
            "axes.titlesize": FONT_SIZE_PT,
            "xtick.labelsize": FONT_SIZE_PT,
            "ytick.labelsize": FONT_SIZE_PT,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    cmap = plt.get_cmap("plasma")
    palette = {
        method: cmap(COLORMAP_POSITIONS[METHOD_ORDER.index(method)])
        for method in methods
    }

    fig, (ax_curve, ax_cases) = plt.subplots(
        1,
        2,
        figsize=(FIGURE_WIDTH_MM / 25.4, FIGURE_HEIGHT_MM / 25.4),
        gridspec_kw={"width_ratios": [1.18, 1.0]},
    )
    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.16, top=0.82, wspace=0.3)

    # (a) Each line is the arithmetic mean of the case-level ranked profiles.
    handles = []
    for method in methods:
        d = curves.loc[curves["experiment"] == method].sort_values("capacity_percentile")
        ax_curve.plot(
            d["capacity_percentile"],
            d["value"],
            lw=2.35,
            color=palette[method],
            solid_capstyle="round",
        )
        handles.append(Line2D([0], [0], color=palette[method], lw=2.8, label=METHOD_LABELS.get(method, method)))

    ax_curve.set_title("(a) Marginal injection utilisation", loc="left", weight="bold", pad=8)
    ax_curve.set_xlabel("Installed injection capacity (%)", labelpad=6)
    ax_curve.set_ylabel("Full-load hours/year", labelpad=7)
    ax_curve.set_xlim(0, 100)
    ax_curve.set_xticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
    ax_curve.set_ylim(bottom=0)
    ax_curve.grid(axis="y", color="0.85", lw=0.55, zorder=0)

    # (b) Every point is a country--weather model case, before aggregation.
    # Deterministic horizontal offsets separate repeated zero values without
    # artificially jittering their actual y-values.
    for i, method in enumerate(methods):
        d = unused.loc[unused["experiment"] == method].copy()
        d["country_rank"] = d["country"].map(
            {country: rank for rank, country in enumerate(countries)}
        )
        d = d.sort_values(["country_rank", "weather"])
        if d.empty:
            continue
        x = i + np.linspace(-0.29, 0.29, len(d))
        y = d["unused_pct"].to_numpy()
        ax_cases.scatter(
            x, y, s=19, color=palette[method], edgecolors="white",
            linewidths=0.35, zorder=3,
        )
        bound = d["at_wtir_bound"].to_numpy(dtype=bool)
        if bound.any():
            ax_cases.scatter(
                x[bound], y[bound], s=48, facecolors="none",
                edgecolors="black", linewidths=0.85, zorder=4,
            )
        median_unused = float(np.median(y))

        # ax_cases.hlines(
        #     y=median_unused,
        #     xmin=i - 0.24,
        #     xmax=i + 0.24,
        #     color="black",
        #     linewidth=1.5,
        #     zorder=6,
        # )

        # ax_cases.annotate(
        #     f"{median_unused:.1f}%",
        #     (i, median_unused),
        #     xytext=(7, 7),
        #     textcoords="offset points",
        #     ha="left",
        #     va="bottom",
        #     fontsize=7.4,
        # )

    ax_cases.set_title("(b) Unused injection capacity", loc="left", weight="bold", pad=8)
    ax_cases.set_xlabel("Formulation", labelpad=6)
    ax_cases.set_ylabel("Unused installed capacity (%)", labelpad=7)
    ax_cases.set_xlim(-0.49, len(methods) - 0.40)
    ax_cases.set_ylim(-5, 100)
    ax_cases.set_xticks(range(len(methods)), [METHOD_LABELS.get(m, m) for m in methods])
    ax_cases.set_yticks([0, 25, 50, 75, 100])
    ax_cases.grid(axis="y", color="0.85", lw=0.55, zorder=0)

    ring_key = Line2D(
        [0], [0], marker="o", linestyle="none", markerfacecolor="none",
        markeredgecolor="black", markeredgewidth=1.0, markersize=8,
        label=rf"WTIR = {wtir_bound:g} (bound)",
    )
    mean_key = Line2D(
        [0], [0], marker="D", linestyle="none", markerfacecolor="black",
        markeredgecolor="white", markeredgewidth=0.6, markersize=6,
        label="Case mean",
    )
    # ax_cases.legend(handles=[ring_key], loc="upper right", frameon=False,
    #                 fontsize=7.8, handletextpad=0.5, borderpad=0.1)
    handles.append(ring_key)
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.999),
        ncol=len(handles)+1, frameon=False, columnspacing=2.6, handlelength=3.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="pdf", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-case", type=Path, required=True,
                        help="CSV exported by plot_marginal_capacity_utilisation")
    parser.add_argument("--design", type=Path, required=True,
                        help="storage_design_outcomes_*_points.csv containing WTIR")
    parser.add_argument("--output", type=Path,
                        default=Path("injection_capacity_utilisation_SC.pdf"))
    parser.add_argument("--technology", default="salt_cavern")
    parser.add_argument("--experiments", nargs="+", default=list(METHOD_ORDER),
                        choices=list(METHOD_LABELS))
    parser.add_argument("--countries", nargs="+", default=list(COUNTRY_ORDER))
    parser.add_argument("--wtir-bound", type=float, default=WTIR_BOUND)
    args = parser.parse_args()

    methods = list(dict.fromkeys(args.experiments))
    countries = list(dict.fromkeys(args.countries))
    cases = load_case_profiles(args.per_case, args.technology, methods, countries)
    designs = load_designs(args.design, args.technology, methods, countries)
    curves, unused = build_results(cases, designs, args.wtir_bound)
    draw_figure(curves, unused, methods, countries, args.wtir_bound, args.output)

    print(f"Saved: {args.output}")
    summary = unused.groupby("experiment", sort=False).agg(
        cases=("unused_pct", "size"),
        mean_unused_pct=("unused_pct", "mean"),
        cases_with_unused=("unused_pct", lambda x: int((x > 0).sum())),
        cases_at_wtir_bound=("at_wtir_bound", "sum"),
    ).reindex(methods)
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
