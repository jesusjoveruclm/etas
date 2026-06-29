import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

import psutil

BENCHMARK_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCHMARK_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from etas.inversion import ETASParameterCalculation
from etas.simulation import ETASSimulation


DEFAULT_N_SIMULATIONS = 10_000
DEFAULT_CHUNKSIZES = [100, 1_000, 10_000]
DEFAULT_CONFIG_PATH = REPO_ROOT / "bench_of_changes" / "sample_config.json"
DEFAULT_PARAMETERS_PATH = REPO_ROOT / "bench_of_changes" / "parameters_ch.json"
DEFAULT_RESULTS_CSV = BENCHMARK_DIR / "benchmark_concat_vs_chunks_results.csv"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "benchmark_concat_vs_chunks_outputs"
MODES = [
    ("list_chunks", True),
    ("dataframe_concat", False),
]
RESULT_COLUMNS = [
    "mode",
    "use_list_chunks",
    "chunksize",
    "n_simulations",
    "elapsed_seconds",
    "seconds_per_simulation",
    "peak_rss_mb",
    "rss_start_mb",
    "rss_end_mb",
    "output_size_mb",
    "output_csv",
    "started_at",
    "ended_at",
    "success",
    "error",
]


def _mb(n_bytes):
    return n_bytes / 1024 / 1024


def _monitor_peak_rss(stop_event, interval_seconds=0.05):
    process = psutil.Process(os.getpid())
    peak = process.memory_info().rss
    while not stop_event.is_set():
        peak = max(peak, process.memory_info().rss)
        time.sleep(interval_seconds)
    peak = max(peak, process.memory_info().rss)
    return peak


def _run_memory_monitor(stop_event, peak_holder):
    peak_holder["peak_rss"] = _monitor_peak_rss(stop_event)


def _load_simulation(config_path, parameters_path):
    with open(config_path, "r") as f:
        forecast_config = json.load(f)
    with open(parameters_path, "r") as f:
        inversion_params = json.load(f)

    etas_invert = ETASParameterCalculation.load_calculation(inversion_params)
    simulation = ETASSimulation(
        etas_invert,
        m_max=forecast_config.get("m_max", None),
    )
    simulation.prepare()
    return simulation, forecast_config


def run_benchmark(job):
    mode, use_list_chunks, chunksize, args_dict = job
    config_path = args_dict["config_path"]
    parameters_path = args_dict["parameters_path"]
    output_dir = Path(args_dict["output_dir"])
    keep_outputs = args_dict["keep_outputs"]
    n_simulations = args_dict["n_simulations"]

    output_csv = output_dir / (
        f"simulations_{mode}_chunksize_{chunksize}_"
        f"n_{n_simulations}.csv"
    )
    output_csv.unlink(missing_ok=True)

    process = psutil.Process(os.getpid())
    started_at = dt.datetime.now()
    rss_start = process.memory_info().rss
    peak_holder = {"peak_rss": rss_start}
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=_run_memory_monitor,
        args=(stop_event, peak_holder),
        daemon=True,
    )

    error = ""
    success = False
    elapsed_seconds = None
    output_size_mb = 0.0

    monitor.start()
    t0 = time.perf_counter()
    try:
        simulation, forecast_config = _load_simulation(
            config_path,
            parameters_path,
        )
        simulation.simulate_to_csv(
            str(output_csv),
            forecast_config["forecast_duration"],
            n_simulations,
            chunksize=chunksize,
            use_list_chunks=use_list_chunks,
        )
        elapsed_seconds = time.perf_counter() - t0
        output_size_mb = _mb(output_csv.stat().st_size)
        success = True
    except Exception:
        elapsed_seconds = time.perf_counter() - t0
        error = traceback.format_exc()
    finally:
        stop_event.set()
        monitor.join()
        rss_end = process.memory_info().rss
        ended_at = dt.datetime.now()

        if output_csv.exists() and not keep_outputs:
            output_csv.unlink()

    return {
        "mode": mode,
        "use_list_chunks": use_list_chunks,
        "chunksize": chunksize,
        "n_simulations": n_simulations,
        "elapsed_seconds": elapsed_seconds,
        "seconds_per_simulation": (
            elapsed_seconds / n_simulations
            if elapsed_seconds is not None else None
        ),
        "peak_rss_mb": _mb(peak_holder["peak_rss"]),
        "rss_start_mb": _mb(rss_start),
        "rss_end_mb": _mb(rss_end),
        "output_size_mb": output_size_mb,
        "output_csv": str(output_csv),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "success": success,
        "error": error,
    }


def write_results(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ETASSimulation.simulate_to_csv using incremental "
            "DataFrame concat versus list chunks."
        )
    )
    parser.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Remove the output directory before running.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_path = DEFAULT_RESULTS_CSV
    output_dir = DEFAULT_OUTPUT_DIR
    chunksizes = DEFAULT_CHUNKSIZES
    max_workers = min(3, len(chunksizes) * len(MODES), os.cpu_count() or 1)

    if args.clean_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args_dict = {
        "config_path": str(DEFAULT_CONFIG_PATH),
        "parameters_path": str(DEFAULT_PARAMETERS_PATH),
        "output_dir": str(output_dir),
        "keep_outputs": False,
        "n_simulations": DEFAULT_N_SIMULATIONS,
    }
    jobs = [
        (mode, use_list_chunks, chunksize, args_dict)
        for chunksize in chunksizes
        for mode, use_list_chunks in MODES
    ]

    rows = []
    print(
        f"Running {len(jobs)} benchmarks with max_workers={max_workers}. "
        f"Results: {results_path}"
    )
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers
    ) as executor:
        future_to_job = {
            executor.submit(run_benchmark, job): job
            for job in jobs
        }
        for future in concurrent.futures.as_completed(future_to_job):
            mode, _, chunksize, _ = future_to_job[future]
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda r: (r["chunksize"], r["mode"]))
            write_results(results_path, rows)
            print(
                f"{mode}, chunksize={chunksize}: "
                f"{row['elapsed_seconds']:.3f}s, "
                f"{row['seconds_per_simulation']:.6f}s/sim, "
                f"peak_rss={row['peak_rss_mb']:.1f} MB, "
                f"success={row['success']}"
            )

    rows.sort(key=lambda r: (r["chunksize"], r["mode"]))
    write_results(results_path, rows)
    print(f"Done. Wrote {results_path}")


if __name__ == "__main__":
    main()
