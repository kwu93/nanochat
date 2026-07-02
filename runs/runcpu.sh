#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# This script was last updated/tuned on Jan 17, 2026.

# Run as:
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# Think of this run as educational/fun demo, not something you should expect to work well.
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.

# all the setup stuff
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# train a small 4 layer model
# I tuned this run to complete in about 30 minutes on my MacBook Pro M3 Max.
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# SFT (~10 minutes on my MacBook Pro M3 Max)
# Two settings differ from a GPU run and matter on CPU/MPS:
# - max-seq-len=2048 (with device-batch-size=8 to keep the 16384-token budget): the SFT packer
#   never crops conversations, and the median chat conversation is ~880 tokens, so a 512 context
#   leaves most conversations unplaceable. The buffer then fills with too-long conversations and
#   every batch becomes all-padding => all targets masked => nan loss => zero gradients (no learning).
# - eval-every=-1: the in-training bpb eval runs the torch.compile'd model, which poisons subsequent
#   training on MPS (loss goes nan from ~step 3). torch.compile can't just be disabled (the fused
#   AdamW optimizer depends on it), so we skip the in-training eval and evaluate separately below.
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft \
    --max-seq-len=2048 \
    --device-batch-size=8 \
    --total-batch-size=16384 \
    --eval-every=-1 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --chatcore-every=-1 \
    --run=$WANDB_RUN

# Evaluate the SFT model on the chat benchmarks (ARC/MMLU/GSM8K/etc).
# Done here instead of in-training because --eval-every=-1 disabled the in-training eval above.
# python -m scripts.chat_eval -i sft

# Chat with the model over CLI
# The model should be able to say that it is Paris.
# It might even know that the color of the sky is blue.
# Sometimes the model likes it if you first say Hi before you ask it questions.
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web
