"""Targeted single-model operation figures: normalized SoC and LIFO residence CDF.

Run from the project root next to plot_storage_residence_profiles.py::

    pixi run python -m scripts.results_analysis.plot_storage_operations `
        --country NLD --experiment FX1 --weather 2018-2021

The case selector accepts 2018, 2018_3y, weather_2018_3y, or 2018-2021.
The default selector is NLD / FX1 / 2018-2021; --list-cases lists models.

Uses exactly the same store loading, LIFO ledger and capacity definitions as
plot_storage_residence_profiles.py (which must be in the same directory).
Battery 2h/4h/8h, CAES and geological storage are included when installed.
Optional --include-hydro adds fixed hydrostorage if SoC/flows are compatible.

LIFO ages are accounting-based: injection precedes withdrawal within each
step and cyclic operation is replayed to resolve previous inventory age.
A physical mixing/residence model is NOT claimed. The CSV records unknown-age
throughput and convergence, and plots use known-age throughput only.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

try:
    from .plot_storage_residence_profiles import (
        DEFAULT_INPUT_DIR,
        FAMILIES,
        LABELS,
        MM,
        PROJECT_ROOT,
        _read_design_csv,
        load_stores,
        matching_cases,
        summarise,
    )
except ImportError:
    from plot_storage_residence_profiles import (
        DEFAULT_INPUT_DIR,
        FAMILIES,
        LABELS,
        MM,
        PROJECT_ROOT,
        _read_design_csv,
        load_stores,
        matching_cases,
        summarise,
    )

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "storage_operations"
GEOLOGICAL = ("salt_cavern", "aquifer", "depleted_gas_field", "lined_rock_cavern")
OTHER = ("battery_2h", "battery_4h", "battery_8h", "compressed_air_storage", "hydrostorage")


def plot_soc(grouped: dict, colours: dict, path: Path, *, svg: bool = False) -> None:
    """Chronology of normalized stored energy, separating geological/other stores."""
    rows = [("Geological storage", GEOLOGICAL), ("Short-duration / existing", OTHER)]
    # Do not draw empty rows, but retain two panels when both groups selected.
    active = [(name, families) for name, families in rows
              if any(family in grouped for family in families)]
    if not active:
        raise ValueError("No installed technologies in selected case")
    height_mm = 110 if len(active)==2 else 80
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(len(active), 1, sharex=True,
                                 figsize=(190*MM, height_mm*MM), squeeze=False)
        fig.subplots_adjust(left=.095, right=.985, top=.94, bottom=.13,
                            hspace=.42 if len(active)>1 else 0)
        for i, (group_label, families) in enumerate(active):
            ax=axes[i,0]
            installed = 0
            for family in families:
                if family not in grouped:
                    continue
                members, _, _ = grouped[family]
                cap = sum(s.capacity for s in members)
                time = members[0].time
                if any(not s.time.equals(time) for s in members):
                    raise ValueError(f"{family}: variants do not share timestamps")
                soc = sum((s.soc for s in members), np.zeros(len(time))) / cap
                ax.plot(time,soc,label=LABELS[family],color=colours[family],linewidth=1.05)
                installed += 1
            ax.set_ylim(-.03, 1.04)
            ax.set_yticks((0,.25,.5,.75,1))
            ax.grid(axis="y", color="0.88", linewidth=.55)
            ax.set_axisbelow(True)
            ax.spines[["top","right"]].set_visible(False)
            ax.set_title(group_label, fontsize=9, weight="bold", loc="left",pad=6)
            ax.legend(loc="lower right",ncol=min(5,installed),frameon=False,
                      fontsize=8.5,handlelength=1.6,columnspacing=1)
        fig.supylabel("State of charge / installed capacity",x=.013,fontsize=9)
        axes[-1,0].set_xlabel("Time",fontsize=9)
        locator=mdates.AutoDateLocator(minticks=4,maxticks=9)
        axes[-1,0].xaxis.set_major_locator(locator)
        axes[-1,0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.savefig(path)
        if svg: fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)


def plot_residence_cdf(grouped: dict, summary: pd.DataFrame,
                       colours: dict, path: Path, *, svg: bool = False) -> None:
    """Per-case energy-weighted LIFO residence CDF and compact statistics."""
    installed = [f for f in FAMILIES if f in grouped]
    n=len(installed)
    if not n:
        raise ValueError("No installed storages to plot")
    height_mm = max(95, 40 + 12.4*n)
    with plt.rc_context({"font.size":9, "pdf.fonttype":42, "svg.fonttype":"none"}):
        fig, ax=plt.subplots(figsize=(190*MM,height_mm*MM))
        fig.subplots_adjust(left=.10,right=.68,top=.92,bottom=.17)
        max_days=1.0
        has_data=False
        for family in installed:
            _, ages, energies=grouped[family]
            if len(ages)==0 or float(energies.sum())<=0:
                continue
            order=np.argsort(ages,kind="stable")
            days=ages[order]/24
            cumulative=np.cumsum(energies[order])/float(energies.sum())
            ax.step(np.r_[0,days],np.r_[0,cumulative],where="post",
                    color=colours[family],linewidth=1.35,label=LABELS[family])
            max_days=max(max_days,float(days[-1]))
            has_data=True
        ax.set_xscale("symlog",linthresh=1)
        ax.set_xlim(0,max_days*1.12)
        ax.set_ylim(0,1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
        ax.set_xlabel("LIFO residence time before withdrawal (days)")
        ax.set_ylabel("Cumulative share of known-age discharge")
        ax.grid(axis="both",color="0.88",linewidth=.55)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
        # if has_data:
        #     ax.legend(loc="upper left",frameon=False,fontsize=8.3,
        #               ncol=2,columnspacing=1,handlelength=1.5)
        # else:
        #     ax.text(.5,.5,"No attributable discharge",ha="center",va="center",
        #             color="0.5",transform=ax.transAxes)
        # A compact reference table to the right; numeric values from this case.
        fig.text(.718,.9,"Technology",fontsize=8.4,weight="bold")
        fig.text(.718,.87,"Residence / design",fontsize=8,color="0.45")
        for i, family in enumerate(installed):
            row=summary.loc[summary["technology"]==family].iloc[0]
            top=.8-i*(.64/max(n,1))
            fig.text(.718,top,LABELS[family],color=colours[family],weight="bold",fontsize=8.8)
            mean=row["mean_term_hours"] / 24
            p50=row["median_term_hours"] / 24
            def fmt(v,units=""):
                return f"{v:.3g}{units}" if math.isfinite(v) else "n/a"
            fig.text(.718,top-.035, f"Mean {fmt(mean,' d')}  |  P50 {fmt(p50,' d')}",
                     fontsize=8.0)
            epr=row["design_EPR_hours"]
            wtir=row["design_WTIR"]
            if math.isfinite(epr) or math.isfinite(wtir):
                fig.text(.718,top-.07,
                         f"EPR {fmt(epr,' h')}  |  WTIR {fmt(wtir)}",fontsize=7.7)
        fig.savefig(path)
        if svg: fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",type=Path,default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--country",default="NLD")
    parser.add_argument("--experiment",default="FX1",help="Exact experiment, e.g. FX1")
    parser.add_argument("--weather",default="2018-2021",help="E.g. 2018-2021")
    parser.add_argument("--include-hydro",action="store_true")
    parser.add_argument("--design-csv",type=Path)
    parser.add_argument("--lifo-max-cycles",type=int,default=8)
    parser.add_argument("--svg",action="store_true")
    parser.add_argument("--list-cases",action="store_true")
    args=parser.parse_args()
    if args.lifo_max_cycles<1:
        parser.error("--lifo-max-cycles must be at least 1")
    matches=matching_cases(args.input_dir,
                           None if args.list_cases else args.country,
                           None if args.list_cases else args.experiment,
                           None if args.list_cases else args.weather)
    if args.list_cases:
        for p,c in matches:
            print(c["country"],c["experiment"],c["weather"],p.name)
        return
    if len(matches)!=1:
        parser.error(f"Selector matched {len(matches)} cases (expected one). "
                     "Use --list-cases and specify country/experiment/weather.")
    path,case=matches[0]
    stores=load_stores(path,include_hydro=args.include_hydro)
    design=_read_design_csv(args.design_csv,case)
    grouped,summary,colours=summarise(stores,case,design,lifo_max_cycles=args.lifo_max_cycles)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    suffix=f"{case['country']}_{case['experiment']}_{case['weather']}"
    soc=args.output_dir/f"storage_soc_{suffix}.pdf"
    residence=args.output_dir/f"storage_lifo_term_{suffix}.pdf"
    audit=args.output_dir/f"storage_lifo_summary_{suffix}.csv"
    plot_soc(grouped,colours,soc,svg=args.svg)
    plot_residence_cdf(grouped,summary,colours,residence,svg=args.svg)
    summary.to_csv(audit,index=False)
    for p in (soc,residence,audit): print("Saved",p)
    if args.svg:
        for p in (soc,residence): print("Saved",p.with_suffix(".svg"))


if __name__ == "__main__":
    main()
