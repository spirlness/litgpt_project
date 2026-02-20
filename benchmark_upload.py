import time
from unittest.mock import MagicMock, patch
import run_train
import concurrent.futures
import huggingface_hub # Ensure this is imported for patching

class MockHfApi:
    def upload_folder(self, **kwargs):
        print("Mock upload started...")
        time.sleep(2.0) # Simulate 2 seconds upload
        print("Mock upload finished.")

def benchmark():
    print("Benchmarking save_checkpoint...")

    mock_fabric = MagicMock()
    mock_fabric.is_global_zero = True

    # Ensure previous executor is cleared
    run_train._shutdown_upload_executor()

    # Patch HfApi in huggingface_hub module, which is where run_train imports it from locally
    with patch("huggingface_hub.HfApi", side_effect=MockHfApi), \
         patch("run_train.Path") as mock_path:

        start = time.time()
        run_train.save_checkpoint(
            fabric=mock_fabric,
            out_dir=MagicMock(),
            step=1,
            total_tokens=100,
            model=MagicMock(),
            optimizer=MagicMock(),
            upload_to_hf=True,
            hf_repo_id="test/repo"
        )
        end = time.time()

        duration = end - start
        print(f"save_checkpoint returned in {duration:.4f} seconds")

        if duration > 1.0:
            print("FAIL: save_checkpoint blocked for too long.")
        else:
            print("PASS: save_checkpoint returned quickly.")

        # Now wait for the executor to finish
        print("Waiting for background upload...")
        executor = run_train._get_upload_executor()
        executor.shutdown(wait=True)
        print("Executor shutdown complete.")

if __name__ == "__main__":
    benchmark()
