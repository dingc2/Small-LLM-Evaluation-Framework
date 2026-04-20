#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# Path shim — makes `python merge_results.py` work from inside sLLM_eval_framework/
# ---------------------------------------------------------------------------
from __future__ import annotations
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# ---------------------------------------------------------------------------

"""
Merge multiple runner result JSON files into a single aggregated_results.json.

Allows incremental model runs: run a few models today, a few more tomorrow,
then merge all results into one file for analysis.

Usage (run from inside sLLM_eval_framework/)
-----
    # Merge all *_results.json files in results/ directory
    python merge_results.py results/

    # Merge specific files
    python merge_results.py results/20260412T162611Z_results.json results/20260412T195501Z_results.json

    # Keep all runs (no dedup — useful to accumulate more repetitions)
    python merge_results.py results/ --strategy all

    # Custom output path
    python merge_results.py results/ --output results/combined.json

Deduplication strategies
------------------------
  latest (default): For the same (model, benchmark, skill_config), keep runs from
      the newest source file only. Useful when you re-run a model to get better results.
  all: Keep every run from every file. Scores are averaged across all runs,
      so this accumulates more repetitions over time.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_file(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _extract_timestamp(run_id: str) -> str:
    """Return a sortable timestamp string from a run_id like '20260412T162611Z'."""
    # run_id is already a sortable ISO-ish string — return as-is
    return run_id


def merge_result_files(
    files: list[Path],
    strategy: Literal["latest", "all"] = "latest",
) -> dict[str, Any]:
    """
    Merge multiple runner result JSONs into one aggregated result dict.

    The output has the same shape as a single run's JSON (run_id, results,
    comparison_table, …) so it can be passed directly to analyze.py.

    Parameters
    ----------
    files:
        Paths to *_results.json files to merge. Must be non-empty.
    strategy:
        "latest" — for each (model, benchmark, skill_config), keep runs only
            from the newest source file.
        "all" — keep runs from every file; scoring averages across all runs.
    """
    if not files:
        raise ValueError("No result files provided.")

    # Sort files by run_id (timestamp) so "latest" can be determined
    loaded: list[tuple[str, dict[str, Any]]] = []
    for fp in files:
        data = _load_file(fp)
        run_id = data.get("run_id", fp.stem)
        loaded.append((run_id, data))
    loaded.sort(key=lambda x: x[0])  # ascending; last = newest

    # Collect all BenchmarkResult dicts, tagged with their source run_id
    # Structure: {(model, benchmark, skill_config_name): [(run_id, result_dict), ...]}
    from collections import defaultdict
    grouped: dict[tuple[str, str, str], list[tuple[str, dict]]] = defaultdict(list)

    for run_id, data in loaded:
        for result in data.get("results", []):
            sc_name = result.get("metadata", {}).get("skill_config_name", "unknown")
            key = (result["model_name"], result["benchmark_name"], sc_name)
            grouped[key].append((run_id, result))

    # Apply dedup strategy
    merged_results: list[dict[str, Any]] = []
    if strategy == "latest":
        for key, entries in grouped.items():
            # Keep only entries from the newest run_id that has data for this key
            newest_run_id = max(run_id for run_id, _ in entries)
            kept = [r for run_id, r in entries if run_id == newest_run_id]
            merged_results.extend(kept)
    else:  # "all"
        for entries in grouped.values():
            merged_results.extend(r for _, r in entries)

    # Reconstruct BenchmarkResult Pydantic objects so we can reuse the
    # comparison table builder from runner.py
    from sLLM_eval_framework.benchmarks.base import BenchmarkResult
    result_objects = [BenchmarkResult.model_validate(r) for r in merged_results]

    from sLLM_eval_framework.runner import _build_comparison_table
    comparison_table = _build_comparison_table(result_objects)

    # Build merged metadata
    all_run_ids = sorted(set(run_id for run_id, _ in loaded))
    n_models = len(set(r["model_name"] for r in merged_results))
    now = datetime.now(timezone.utc).isoformat()

    return {
        "run_id": f"aggregated_{all_run_ids[-1]}",
        "started_at": loaded[0][1].get("started_at", now),
        "finished_at": loaded[-1][1].get("finished_at", now),
        "duration_s": sum(d.get("duration_s", 0) for _, d in loaded),
        "config": {
            "note": "Aggregated from multiple runs",
            "source_run_ids": all_run_ids,
            "strategy": strategy,
            "n_source_files": len(files),
            "n_models": n_models,
        },
        "results": merged_results,
        "comparison_table": comparison_table,
    }


def auto_merge(output_dir: Path, strategy: Literal["latest", "all"] = "latest") -> Path | None:
    """
    Merge all *_results.json in output_dir (excluding aggregated_results.json)
    and write aggregated_results.json.  Called automatically by the runner.

    Returns the output path, or None if there are fewer than 2 source files.
    """
    source_files = sorted(
        p for p in output_dir.glob("*_results.json")
        if p.name != "aggregated_results.json"
    )
    if len(source_files) < 1:
        return None

    out_path = output_dir / "aggregated_results.json"
    merged = merge_result_files(source_files, strategy=strategy)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, default=str)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple runner result JSON files into aggregated_results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more *_results.json files, OR a directory containing them.",
    )
    parser.add_argument(
        "--strategy",
        choices=["latest", "all"],
        default="latest",
        help=(
            "Dedup strategy. 'latest': for each model+benchmark, keep only runs "
            "from the newest file. 'all': keep every run from every file. "
            "(default: latest)"
        ),
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the merged JSON (default: <first_input_dir>/aggregated_results.json)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing a companion CSV summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Resolve input files
    files: list[Path] = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            found = sorted(
                f for f in p.glob("*_results.json")
                if f.name != "aggregated_results.json"
            )
            if not found:
                print(f"Warning: no *_results.json files found in {p}")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: {p} does not exist, skipping.")

    if not files:
        print("Error: no result files found.")
        sys.exit(1)

    print(f"Merging {len(files)} file(s) with strategy='{args.strategy}':")
    for f in files:
        print(f"  {f}")

    merged = merge_result_files(files, strategy=args.strategy)

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = files[0].parent / "aggregated_results.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2, default=str)
    print(f"\nSaved merged results → {out_path}")

    # Print comparison table summary
    table = merged.get("comparison_table", [])
    if table:
        models = sorted(set(r["model"] for r in table))
        print(f"\nCovered {len(models)} model(s): {', '.join(models)}")
        print(f"Comparison table: {len(table)} row(s)")

    # Optionally write CSV
    if not args.no_csv and table:
        import csv
        csv_path = out_path.with_suffix(".csv")
        fieldnames = list(table[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(table)
        print(f"Saved companion CSV  → {csv_path}")


if __name__ == "__main__":
    main()
