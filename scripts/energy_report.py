"""Energy used by all IDUN jobs of the project, by kind of work.

GPU energy is measured: jobs/gpu.slurm logs the job's GPU power draw every
30 s, and a job's energy is its mean power times its run time. CPU energy is
estimated from allocated core-hours (Slurm does not meter it); IDUN's CPU
nodes are dual Intel Xeon (48-56 cores, about 150-205 W per socket), so
--watts-per-core 7 by default. Data-centre overhead (cooling, PUE) is left out.

    sacct -X -P -n -S 2026-11-01 -o JobID,JobName,Partition,State,ElapsedRaw,AllocTRES > sacct.txt
    python scripts/energy_report.py sacct.txt --gpu-logs logs ~/tdt17-football-tracking/logs

Every job counts, including failed and cancelled ones: they were part of the
experimenting too.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

KINDS = [  # (kind, job name pattern), first match wins
    ("training", r"^(d\d|rfdetr-m|final-m)"),
    ("detection", r"^(det-|val-d6)"),
    ("tracking and evaluation", r"^(track|tune-)"),
    ("jersey reading", r"^(jersey-|bench-)"),
    ("dataset building", r"^yolo-"),
    ("rendering videos", r"^render-"),
]
EV_KWH_PER_KM = 0.17  # a typical electric car, 17 kWh per 100 km


def kind(name: str) -> str:
    return next((k for k, pattern in KINDS if re.match(pattern, name)), "other")


def gpu_power(logs: list[Path]) -> dict[str, float]:
    """Mean GPU power (W) per job ID, from the nvidia-smi logs."""
    power = {}
    for directory in logs:
        for path in directory.glob("*-gpu.csv"):
            job = path.name.removesuffix("-gpu.csv").rsplit("-", 1)[1]
            df = pd.read_csv(path, skipinitialspace=True)
            watts = pd.to_numeric(df["power.draw [W]"].astype(str).str.removesuffix(" W"), errors="coerce").dropna()
            if len(watts):
                power[job] = watts.mean()
    return power


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sacct", type=Path, help="sacct -X -P output, see above")
    parser.add_argument("--gpu-logs", type=Path, nargs="+", required=True, help="folders with *-gpu.csv files")
    parser.add_argument("--watts-per-core", type=float, default=7.0)
    args = parser.parse_args()

    jobs = pd.read_csv(args.sacct, sep="|", header=None,
                       names=["job", "name", "partition", "state", "seconds", "tres"], dtype={"job": str})
    jobs["hours"] = jobs["seconds"] / 3600
    jobs["cores"] = jobs["tres"].str.extract(r"cpu=(\d+)", expand=False).astype(float)
    jobs["gpus"] = jobs["tres"].str.extract(r"gres/gpu=(\d+)", expand=False).astype(float).fillna(0)
    jobs["kind"] = jobs["name"].map(kind)

    power = gpu_power(args.gpu_logs)
    jobs["gpu_watts"] = jobs["job"].map(power)
    gpu_jobs = jobs["gpus"] > 0
    unlogged = gpu_jobs & jobs["gpu_watts"].isna() & (jobs["seconds"] > 0)
    fallback = (jobs.loc[gpu_jobs, "gpu_watts"] * jobs.loc[gpu_jobs, "seconds"]).sum() / \
        jobs.loc[gpu_jobs & jobs["gpu_watts"].notna(), "seconds"].sum()  # time-weighted mean power
    jobs.loc[unlogged, "gpu_watts"] = fallback
    jobs["gpu_kwh"] = jobs["gpu_watts"].fillna(0) * jobs["gpus"] * jobs["hours"] / 1000
    jobs["cpu_kwh"] = jobs["cores"] * args.watts_per_core * jobs["hours"] / 1000
    jobs["gpu_hours"] = jobs["gpus"] * jobs["hours"]
    jobs["core_hours"] = jobs["cores"] * jobs["hours"]

    table = jobs.groupby("kind").agg(jobs=("job", "size"), gpu_hours=("gpu_hours", "sum"), gpu_kwh=("gpu_kwh", "sum"),
                                     core_hours=("core_hours", "sum"), cpu_kwh=("cpu_kwh", "sum"))
    table["kwh"] = table["gpu_kwh"] + table["cpu_kwh"]
    table = table.sort_values("kwh", ascending=False)
    table.loc["total"] = table.sum()
    print(table.round({"gpu_hours": 1, "gpu_kwh": 2, "core_hours": 0, "cpu_kwh": 2, "kwh": 2}).to_string())

    total = table.loc["total", "kwh"]
    print(f"\n{len(jobs)} jobs ({(jobs['state'].str.split().str[0] != 'COMPLETED').sum()} failed, timed out or cancelled); "
          f"{unlogged.sum()} GPU jobs without a power log counted at the mean {fallback:.0f} W")
    print(f"GPU energy measured: {table.loc['total', 'gpu_kwh']:.1f} kWh; CPU energy estimated "
          f"({args.watts_per_core:g} W per core): {table.loc['total', 'cpu_kwh']:.1f} kWh")
    print(f"Total {total:.1f} kWh: about {total / EV_KWH_PER_KM:.0f} km in an electric car "
          f"({EV_KWH_PER_KM * 100:g} kWh per 100 km)")


if __name__ == "__main__":
    main()
