#!/bin/bash

#SBATCH --job-name=uhs_dispatch
#SBATCH --partition=compute-p1
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=1000
#SBATCH --account=research-tpm-ess
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

# Maximum simultaneous optimisations.
MAX_PARALLEL=10

# Format:
# COUNTRY REPRESENTATION WEATHER
EXPERIMENTS=(

    # --------------------------------------------------------
    # 2009–2012
    # --------------------------------------------------------

    # "NLD B1  weather_2009_3y"
    # "NLD P1  weather_2009_3y"
    # "NLD F1  weather_2009_3y"
    # "NLD FX1 weather_2009_3y"

    # --------------------------------------------------------
    # 2012–2015
    # --------------------------------------------------------

    # "NLD B1  weather_2012_3y"
    # "NLD P1  weather_2012_3y"
    # "NLD F1  weather_2012_3y"
    # "NLD FX1 weather_2012_3y"

    # --------------------------------------------------------
    # 2015–2018
    # --------------------------------------------------------

    # "NLD B1  weather_2015_3y"
    # "NLD P1  weather_2015_3y"
    # "NLD F1  weather_2015_3y"
    # "NLD FX1 weather_2015_3y"

    # --------------------------------------------------------
    # 2018–2021
    # --------------------------------------------------------

    # "NLD B1  weather_2018_3y"
    # "NLD P1  weather_2018_3y"
    # "NLD F1  weather_2018_3y"
    # "NLD FX1 weather_2018_3y"

    # --------------------------------------------------------
    # 2021–2024
    # --------------------------------------------------------

    # "NLD B1  weather_2021_3y"
    # "NLD P1  weather_2021_3y"
    # "NLD F1  weather_2021_3y"
    # "NLD FX1 weather_2021_3y"

)

EXPERIMENTS=(

    # --------------------------------------------------------
    # 2009–2012
    # --------------------------------------------------------

    "ESP B1  weather_2009_3y"
    "ESP FX1 weather_2009_3y"

    # --------------------------------------------------------
    # 2012–2015
    # --------------------------------------------------------

    "ESP B1  weather_2012_3y"
    "ESP FX1 weather_2012_3y"

    # --------------------------------------------------------
    # 2015–2018
    # --------------------------------------------------------

    "ESP B1  weather_2015_3y"
    "ESP FX1 weather_2015_3y"

    # --------------------------------------------------------
    # 2018–2021
    # --------------------------------------------------------

    "ESP B1  weather_2018_3y"
    "ESP FX1 weather_2018_3y"

    # --------------------------------------------------------
    # 2021–2024
    # --------------------------------------------------------

    "ESP B1  weather_2021_3y"
    "ESP FX1 weather_2021_3y"

)

# ============================================================
# SUBMISSION
# ============================================================

# When run on the login node, submit the complete job array.
# No Python execution is required.

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then

    mkdir -p logs

    NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
    LAST_INDEX=$((NUM_EXPERIMENTS - 1))

    echo "Submitting ${NUM_EXPERIMENTS} experiments."
    echo "Maximum parallel jobs: ${MAX_PARALLEL}"

    sbatch \
        --array="0-${LAST_INDEX}%${MAX_PARALLEL}" \
        "$0"

    exit 0
fi

# ============================================================
# RUN INDIVIDUAL EXPERIMENT
# ============================================================

EXPERIMENT="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"

read -r COUNTRY REPRESENTATION WEATHER <<< "$EXPERIMENT"

echo "=================================================="
echo "Job ID:         $SLURM_JOB_ID"
echo "Array task:     $SLURM_ARRAY_TASK_ID"
echo "Host:           $(hostname)"
echo "Country:        $COUNTRY"
echo "Representation: $REPRESENTATION"
echo "Weather:        $WEATHER"
echo "CPUs:           $SLURM_CPUS_PER_TASK"
echo "=================================================="

# ============================================================
# ENVIRONMENT
# ============================================================

module load gurobi/12.0.0

export GRB_LICENSE_FILE=/apps/generic/gurobi/12.0.0/linux64/lib/gurobi.lic

# ============================================================
# PROJECT
# ============================================================

cd "$HOME/uhs_representation/uhs_cem_representation"

export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ============================================================
# EXECUTION
# ============================================================

srun pixi run -e delftblue --locked \
    python -m hpc.run_model \
    --country "$COUNTRY" \
    --representation "$REPRESENTATION" \
    --weather "$WEATHER" \
    --threads "$SLURM_CPUS_PER_TASK"

echo "Experiment completed."