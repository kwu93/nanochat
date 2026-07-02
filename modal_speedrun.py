"""
Run the nanochat speedrun pipeline on Modal's GPU cloud.

This mirrors runs/speedrun.sh, but instead of provisioning deps with uv on the
local machine and running torchrun there, it bakes the dependencies into a Modal
image and runs the pipeline on rented H100s. All intermediate artifacts (dataset,
tokenizer, checkpoints, report) live in a persistent Modal Volume mounted at /cache
(NANOCHAT_BASE_DIR), so they survive across runs and pipeline steps.

Usage (from the repo root, after `pip install modal` and `modal token new`):

    # Cheap end-to-end smoke test on 1xH100 (tiny model, a few minutes, ~$1-2):
    modal run modal_speedrun.py --mode smoke

    # Full GPT-2-grade speedrun on 8xH100 (~3h, depth-24 + fp8):
    modal run modal_speedrun.py --mode full

    # With Weights & Biases logging (create the secret once, then set USE_WANDB):
    #   modal secret create wandb WANDB_API_KEY=xxxxxxxx
    USE_WANDB=1 modal run modal_speedrun.py --mode full --wandb-run my-speedrun

Browse artifacts afterwards with:  modal volume ls nanochat-cache
"""

import os
import subprocess

import modal

# -----------------------------------------------------------------------------
# App, persistent volume, and the container image

app = modal.App("nanochat-speedrun")

# Persistent storage for dataset shards, tokenizer, checkpoints and reports.
cache_vol = modal.Volume.from_name("nanochat-cache", create_if_missing=True)

CACHE = "/cache"        # NANOCHAT_BASE_DIR inside the container
REPO = "/root/nanochat"  # where this repo's source is mounted
HF_HOME = f"{CACHE}/huggingface"  # cache HF assets (e.g. the FA3 kernel) on the volume

# Dependencies mirror pyproject.toml. torch comes from the CUDA 12.8 wheel index
# (same as `uv sync --extra gpu`); everything else comes from PyPI. Flash-Attention 3
# is fetched at runtime by the `kernels` package on Hopper GPUs (see
# nanochat/flash_attention.py), so it is not installed here.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "datasets>=4.0.0",
        "fastapi>=0.117.1",
        "kernels>=0.11.7",
        "rustbpe>=0.1.0",
        "tiktoken>=0.11.0",
        "tokenizers>=0.22.0",
        "uvicorn>=0.36.0",
        "wandb>=0.21.3",
        "psutil>=7.1.0",
        "requests>=2.31.0",
    )
    .env({"NANOCHAT_BASE_DIR": CACHE, "OMP_NUM_THREADS": "1", "HF_HOME": HF_HOME})
    .workdir(REPO)
    # Source is added at runtime (copy=False) so editing code doesn't rebuild the image.
    .add_local_dir(
        ".",
        REPO,
        ignore=[".venv", ".git", "**/__pycache__", "*.log", "explore.ipynb"],
    )
)

# Hugging Face token (raises Hub rate limits; used by the FA3 kernel fetch on all ranks).
# Requires: `modal secret create huggingface HF_TOKEN=hf_...`.
HF_SECRET = modal.Secret.from_name("huggingface")
# Attach the wandb secret only when the user opts in (USE_WANDB=1 locally), so the
# default/smoke path needs no wandb secret. Requires: `modal secret create wandb WANDB_API_KEY=...`.
WANDB_SECRETS = [modal.Secret.from_name("wandb")] if os.environ.get("USE_WANDB") else []
SECRETS = [HF_SECRET, *WANDB_SECRETS]

IDENTITY_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"

# -----------------------------------------------------------------------------
# Pipeline (runs inside the GPU container)


def _sh(cmd: list[str]) -> None:
    """Run a command from the repo root, streaming output, raising on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO, check=True)


def _torchrun(nproc: int, module: str, script_args: list[str]) -> None:
    """Launch a training/eval script under torchrun (single node, `nproc` GPUs)."""
    _sh([
        "torchrun", "--standalone", f"--nproc_per_node={nproc}",
        "-m", module, "--", *script_args,
    ])


def run_pipeline(mode: str, wandb_run: str) -> None:
    smoke = mode == "smoke"
    nproc = 1 if smoke else 8
    dataset_shards = 8 if smoke else 170

    # --- Report + tokenizer -------------------------------------------------
    _sh(["python", "-m", "nanochat.report", "reset"])
    _sh(["python", "-m", "nanochat.dataset", "-n", str(dataset_shards)])
    _sh(["python", "-m", "scripts.tok_train", "--max-chars=2000000000", "--vocab-size=32768"])
    _sh(["python", "-m", "scripts.tok_eval"])
    cache_vol.commit()

    # --- Base model pretraining + eval -------------------------------------
    if smoke:
        base_train_args = [
            "--depth=6", "--head-dim=64", "--window-pattern=L", "--max-seq-len=512",
            "--device-batch-size=32", "--total-batch-size=16384",
            "--eval-every=100", "--eval-tokens=524288",
            "--core-metric-every=-1", "--sample-every=100", "--num-iterations=600",
            f"--run={wandb_run}",
        ]
        base_eval_args = ["--device-batch-size=1", "--split-tokens=16384", "--max-per-task=16"]
    else:
        base_train_args = [
            "--depth=24", "--target-param-data-ratio=8", "--device-batch-size=16", "--fp8",
            f"--run={wandb_run}",
        ]
        base_eval_args = ["--device-batch-size=16"]
    _torchrun(nproc, "scripts.base_train", base_train_args)
    _torchrun(nproc, "scripts.base_eval", base_eval_args)
    cache_vol.commit()

    # --- SFT + chat eval ----------------------------------------------------
    _sh(["curl", "-L", "-o", os.path.join(CACHE, "identity_conversations.jsonl"), IDENTITY_URL])
    if smoke:
        sft_args = [
            "--max-seq-len=512", "--device-batch-size=32", "--total-batch-size=16384",
            "--eval-every=200", "--eval-tokens=524288", "--num-iterations=1500",
            f"--run={wandb_run}",
        ]
    else:
        sft_args = ["--device-batch-size=16", f"--run={wandb_run}"]
    _torchrun(nproc, "scripts.chat_sft", sft_args)
    _torchrun(nproc, "scripts.chat_eval", ["-i", "sft"])
    cache_vol.commit()

    # --- Final report -------------------------------------------------------
    _sh(["python", "-m", "nanochat.report", "generate"])
    cache_vol.commit()
    print("\nDone. Artifacts are in the 'nanochat-cache' volume (see the report/ dir).", flush=True)


# -----------------------------------------------------------------------------
# GPU functions


@app.function(
    image=image,
    gpu="H100:1",
    volumes={CACHE: cache_vol},
    timeout=60 * 60,  # 1 hour
    secrets=SECRETS,
)
def train_smoke(wandb_run: str = "dummy") -> None:
    run_pipeline("smoke", wandb_run)


@app.function(
    image=image,
    gpu="H100:8",
    volumes={CACHE: cache_vol},
    timeout=6 * 60 * 60,  # 6 hours
    secrets=SECRETS,
)
def train_full(wandb_run: str = "dummy") -> None:
    run_pipeline("full", wandb_run)


# -----------------------------------------------------------------------------
# Local entrypoint: `modal run modal_speedrun.py --mode {smoke,full}`


@app.local_entrypoint()
def main(mode: str = "smoke", wandb_run: str = "dummy") -> None:
    assert mode in ("smoke", "full"), "mode must be 'smoke' or 'full'"
    if mode == "smoke":
        train_smoke.remote(wandb_run)
    else:
        train_full.remote(wandb_run)
