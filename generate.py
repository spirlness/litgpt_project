import torch
from pathlib import Path
import sys
import time
import yaml
import threading
import queue
from litgpt import GPT, Config
from litgpt.tokenizer import Tokenizer
import argparse

STREAMER_JOIN_TIMEOUT_SECONDS = 1.0  # Small timeout to avoid hanging on shutdown.


class AsyncTokenStreamer:
    def __init__(self):
        self.queue = queue.Queue()
        self.stop_signal = object()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            token = self.queue.get()
            if token is self.stop_signal:
                self.queue.task_done()
                break
            print(token, end="", flush=True)
            self.queue.task_done()

    def put(self, token):
        self.queue.put(token)

    def close(self):
        self.queue.put(self.stop_signal)
        self.thread.join(timeout=STREAMER_JOIN_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            print(
                "Warning: token streamer thread did not exit cleanly within timeout. "
                "This may indicate a threading issue."
            )


def generate(
    checkpoint_dir: Path = Path("checkpoints"),
    prompt: str = "Once upon a time",
    max_new_tokens: int = 200,
    top_k: int = 50,
    temperature: float = 0.8,
    device: str = "auto",
):
    """
    Generate text from a trained model.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    config_path = checkpoint_dir / "litgpt_config.yaml"
    if not config_path.exists():
        config_path = Path("litgpt_config.yaml")

    if not config_path.exists():
        print(
            f"Error: Config not found at {config_path}. Run create_litgpt_config.py first."
        )
        return

    with open(config_path, "r", encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
        model_config_dict = full_config.get("model_config", {})

    # Fix for YAML loading 1e-5 as string
    if "norm_eps" in model_config_dict:
        model_config_dict["norm_eps"] = float(model_config_dict["norm_eps"])

    config = Config(**model_config_dict)

    tokenizer_dir = Path("data/tokenizer")
    if not tokenizer_dir.exists():
        print(
            f"Error: Tokenizer not found at {tokenizer_dir}. Run prepare_data.py first."
        )
        return

    tokenizer = Tokenizer(tokenizer_dir)

    print(f"Loading model architecture: {config.name}")
    model = GPT(config)

    ckpt_path = checkpoint_dir / "final" / "lit_model.pth"

    if not ckpt_path.exists():
        ckpts = list(checkpoint_dir.glob("**/*.pth"))
        if ckpts:
            ckpt_path = sorted(ckpts)[-1]
            print(f"Found checkpoint: {ckpt_path}")
        else:
            print("Warning: No checkpoint found! Generating with random weights.")
            ckpt_path = None

    if ckpt_path:
        print(f"Loading weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    encoded = tokenizer.encode(prompt, device=device)
    prompt_length = encoded.size(0)

    print("\nGenerating...")
    print("-" * 50)

    streamer = AsyncTokenStreamer()
    streamer.put(prompt)

    t0 = time.perf_counter()
    generated_tokens = 0
    for i in range(max_new_tokens):
        if encoded.size(0) > config.block_size:
            idx_cond = encoded[-config.block_size :]
        else:
            idx_cond = encoded

        with torch.no_grad():
            logits = model(idx_cond.unsqueeze(0))

        logits = logits[0, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float("Inf")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        token_str = tokenizer.decode(idx_next)
        streamer.put(token_str)

        encoded = torch.cat((encoded, idx_next))
        generated_tokens += 1

    t1 = time.perf_counter()
    streamer.close()
    print("\n" + "-" * 50)
    elapsed = t1 - t0
    print(f"\nTime for {generated_tokens} tokens: {elapsed:.2f} s")
    if elapsed > 0:
        print(f"Tokens per second: {generated_tokens / elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--max_tokens", type=int, default=200)

    args = parser.parse_args()

    generate(
        checkpoint_dir=Path(args.ckpt_dir),
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
    )
