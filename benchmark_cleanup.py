import tempfile
import time
from pathlib import Path


def setup_checkpoints(base_dir, num_checkpoints=20000):
    print(f"Setting up {num_checkpoints} dummy checkpoints in {base_dir}...")
    start_time = time.time()
    for i in range(num_checkpoints):
        # Create directories step-00000000 to step-00019999
        (base_dir / f"step-{i:08d}").mkdir()
    print(f"Setup complete in {time.time() - start_time:.2f} seconds.")

def benchmark_baseline(out_dir, current_step_dir):
    start_time = time.time()

    # Original logic
    all_checkpoints = sorted(out_dir.glob("step-*"))
    if not all_checkpoints:
        return 0

    deleted_count = 0
    for checkpoint in all_checkpoints:
        if checkpoint.name < current_step_dir.name:
            if checkpoint.is_dir():
                # Simulate deletion (no-op) to isolate logic overhead
                # shutil.rmtree(checkpoint)
                deleted_count += 1

    duration = time.time() - start_time
    return duration, deleted_count

def benchmark_optimized(out_dir, current_step_dir):
    start_time = time.time()

    # Optimized logic: remove sorted()
    # all_checkpoints is now an iterator
    all_checkpoints = out_dir.glob("step-*")

    deleted_count = 0
    # Iterate directly
    for checkpoint in all_checkpoints:
        if checkpoint.name < current_step_dir.name:
            if checkpoint.is_dir():
                # Simulate deletion (no-op)
                # shutil.rmtree(checkpoint)
                deleted_count += 1

    duration = time.time() - start_time
    return duration, deleted_count

def run_benchmark():
    num_checkpoints = 50000
    current_step_index = 40000 # Delete 40,000 checkpoints

    with tempfile.TemporaryDirectory() as tmpdirname:
        out_dir = Path(tmpdirname)
        setup_checkpoints(out_dir, num_checkpoints)

        current_step_dir = out_dir / f"step-{current_step_index:08d}"

        print("\n--- Running Baseline ---")
        baseline_time, baseline_deleted = benchmark_baseline(out_dir, current_step_dir)
        print(f"Baseline Time: {baseline_time:.4f} seconds")
        print(f"Deleted (simulated): {baseline_deleted}")

        print("\n--- Running Optimized ---")
        optimized_time, optimized_deleted = benchmark_optimized(out_dir, current_step_dir)
        print(f"Optimized Time: {optimized_time:.4f} seconds")
        print(f"Deleted (simulated): {optimized_deleted}")

        if baseline_deleted != optimized_deleted:
            print(f"ERROR: Deleted count mismatch! Baseline: {baseline_deleted}, Optimized: {optimized_deleted}")
            exit(1)

        improvement = baseline_time - optimized_time
        percent = (improvement / baseline_time) * 100 if baseline_time > 0 else 0
        print(f"\nImprovement: {improvement:.4f} seconds ({percent:.2f}%)")

if __name__ == "__main__":
    run_benchmark()
