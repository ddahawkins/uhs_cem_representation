"""Run one Calliope scenario, intended for a DelftBlue Slurm array task.

Example:
    python -m scripts.capacity_expansion_models.run_model \
        --country NLD --representation FX1 --weather weather_2015_3y --threads 10

Solver settings remain in config/model.yaml. Only a proven optimal solution
is written to NetCDF; unsuccessful runs exit with a nonzero status.
"""

import argparse
import gc
import os
import time
from pathlib import Path

import calliope

from src.helpers.calliope.load_timeseries import build_calliope_timeseries


calliope.set_log_verbosity("INFO", include_solver_output=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "config" / "model.yaml"
OUTPUT_DIR = PROJECT_ROOT / "hpc" / "results_2h_no_coupling_bounds"

START_DATE = "2009-04-01"
END_DATE = "2024-03-31"
SOLVER = "gurobi"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", required=True)
    parser.add_argument("--representation", required=True)
    parser.add_argument("--weather", required=True)
    parser.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help="Allocated CPU count for logging and threading environment variables. "
             "Set Gurobi Threads in model.yaml to match this allocation.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")

    scenario = f"{args.country},{args.representation},{args.weather}"
    output_path = OUTPUT_DIR / (
        f"{args.country}_{args.representation}_{args.weather}.nc"
    )

    # Never trust an existing file's size alone as proof of optimality.
    # Existing files are treated as previously validated results, so delete or
    # explicitly overwrite any legacy files whose status is uncertain.
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. "
            "Verify it was produced by a proven-optimal run or use --overwrite."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)

    print(f"Scenario: {scenario}", flush=True)
    print(f"Allocated CPUs: {args.threads}", flush=True)
    print(f"Output: {output_path}", flush=True)

    started = time.monotonic()
    model = None
    # Per-job temp file: cannot collide with another Slurm job for the same case.
    temp_path = output_path.with_name(
        f"{output_path.stem}.{os.getpid()}.pending.nc"
    )

    try:
        timeseries = build_calliope_timeseries(
            country=args.country,
            start_date=START_DATE,
            end_date=END_DATE,
        )
        if timeseries.index.tz is not None:
            timeseries.index = timeseries.index.tz_localize(None)

        model = calliope.read_yaml(
            MODEL_PATH,
            scenario=scenario,
            data_table_dfs={"country_timeseries": timeseries},
        )
        print(f"Timesteps: {model.inputs.sizes['timesteps']}", flush=True)

        model.build()
        model.solve(solver=SOLVER)

        # Calliope 0.7 exposes this on model.runtime. Do not accept a feasible
        # incumbent, time_limit, suboptimal or unavailable status.
        termination = getattr(model.runtime, "termination_condition", None)
        termination_text = str(termination).strip().lower() if termination is not None else ""
        print(f"Solver termination: {termination_text or 'UNAVAILABLE'}", flush=True)

        if termination_text != "optimal":
            raise RuntimeError(
                "Gurobi did not prove optimality "
                f"(termination={termination_text or 'unavailable'}). "
                "No NetCDF output was saved."
            )

        if getattr(model, "results", None) is None or not model.results.data_vars:
            raise RuntimeError("Optimal termination, but the results dataset is empty")

        # Only a proven optimum is serialized. Rename atomically on the same
        # filesystem after the complete NetCDF has been written.
        model.to_netcdf(temp_path)
        temp_path.replace(output_path)
        print(f"SAVED: {output_path}", flush=True)
        print(f"Elapsed seconds: {time.monotonic() - started:.1f}", flush=True)
        return 0
    finally:
        # If writing fails or Slurm interrupts cleanly, never retain a pending
        # file that might be mistaken for a result.
        temp_path.unlink(missing_ok=True)
        del model
        gc.collect()


if __name__ == "__main__":
    raise SystemExit(main())
