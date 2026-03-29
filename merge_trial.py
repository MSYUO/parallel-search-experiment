import argparse
import csv
import glob
import json
import os
import sys


def merge_trial(trial: int, n_workers: int, k: int, difficulty: int,
                results_dir: str = "results") -> dict:
    pattern = os.path.join(results_dir, f"result_w*_t{trial}.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"[merge] ERROR: no JSON files found for trial={trial} in '{results_dir}'",
              file=sys.stderr)
        sys.exit(1)

    all_found_times = []
    total_hashes = 0

    for path in files:
        with open(path) as f:
            data = json.load(f)
        all_found_times.extend(data["found_times"])
        total_hashes += data["total_hashes"]
        print(f"[merge] Loaded {path}  "
              f"worker_id={data['worker_id']}  "
              f"found={len(data['found_times'])}  "
              f"elapsed={data['elapsed_seconds']:.3f}s")

    all_found_times.sort()
    print(f"[merge] Combined {len(all_found_times)} partial solutions across "
          f"{len(files)} worker(s), sorted by time")

    if len(all_found_times) < k:
        print(f"[merge] ERROR: need {k} solutions but only found "
              f"{len(all_found_times)}. "
              f"Increase N_WORKERS or K_TARGET is too large.",
              file=sys.stderr)
        sys.exit(1)

    t_real = all_found_times[k - 1]   # k번째(0-indexed: k-1) 발견 시점
    print(f"[merge] T_real = {t_real:.6f}s  "
          f"(k={k}th solution, total_hashes={total_hashes:,})")

    return {
        "trial":               trial,
        "n_workers":           n_workers,
        "k":                   k,
        "difficulty":          difficulty,
        "T_real":              round(t_real, 6),
        "total_hashes_all":    total_hashes,
    }


def append_csv(row: dict, csv_path: str) -> None:
    fieldnames = ["trial", "n_workers", "k", "difficulty", "T_real", "total_hashes_all"]
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[merge] Appended to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge worker JSONs → T_real → CSV")
    parser.add_argument("--trial",      type=int, required=True)
    parser.add_argument("--n-workers",  type=int, required=True)
    parser.add_argument("--k",          type=int, required=True)
    parser.add_argument("--difficulty", type=int, required=True)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--csv",        default="results/experiment_results.csv")
    args = parser.parse_args()

    row = merge_trial(
        trial=args.trial,
        n_workers=args.n_workers,
        k=args.k,
        difficulty=args.difficulty,
        results_dir=args.results_dir,
    )
    append_csv(row, args.csv)


if __name__ == "__main__":
    main()
