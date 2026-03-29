import hashlib
import time
import json
import os

# counter file lives in the shared results volume
_COUNTER_FILE = None   # resolved lazily from RESULTS_DIR


def _resolve_worker_id() -> int:
    """WORKER_ID env var takes priority.
    Otherwise each worker races to atomically create a claim file
    (.claimed_id_1, .claimed_id_2, ...) using O_CREAT|O_EXCL, which is
    atomic on both POSIX (Linux/Docker) and NTFS (Windows).
    The first unclaimed slot wins → IDs are always 1, 2, 3, ..."""
    env = os.environ.get("WORKER_ID")
    if env:
        return int(env)

    results_dir = os.environ.get("RESULTS_DIR", "/results")
    os.makedirs(results_dir, exist_ok=True)

    for candidate in range(1, 1024):
        claim = os.path.join(results_dir, f".claimed_id_{candidate}")
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return candidate
        except FileExistsError:
            continue  # another worker claimed this slot

    raise RuntimeError("Could not claim a worker ID after 1024 attempts")

def run_worker():
    worker_id    = _resolve_worker_id()
    difficulty   = int(os.environ.get("DIFFICULTY",   5))
    k_target     = int(os.environ.get("K_TARGET",     10))
    trial_number = int(os.environ.get("TRIAL_NUMBER", 1))

    seed   = f"worker_{worker_id}_{trial_number}"
    prefix = "0" * difficulty

    found_times   = []
    total_hashes  = 0
    nonce         = 0
    start         = time.perf_counter()

    while len(found_times) < k_target:
        digest = hashlib.sha256(f"{seed}:{nonce}".encode()).hexdigest()
        total_hashes += 1

        if digest.startswith(prefix):
            elapsed = time.perf_counter() - start
            found_times.append(round(elapsed, 6))
            count = len(found_times)
            print(f"[Worker {worker_id}] Found #{count}/{k_target} at {elapsed:.2f}s "
                  f"(nonce={nonce}, hash={digest[:12]}...)",
                  flush=True)

        nonce += 1

    elapsed_total = time.perf_counter() - start

    result = {
        "worker_id":       worker_id,
        "trial":           trial_number,
        "difficulty":      difficulty,
        "k_target":        k_target,
        "seed":            seed,
        "elapsed_seconds": round(elapsed_total, 6),
        "total_hashes":    total_hashes,
        "found_times":     found_times,
    }

    results_dir = os.environ.get("RESULTS_DIR", "/results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, f"result_w{worker_id}_t{trial_number}.json")

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Worker {worker_id}] Done. Total={elapsed_total:.2f}s, "
          f"Hashes={total_hashes:,}, Saved to {out_path}",
          flush=True)

if __name__ == "__main__":
    run_worker()
