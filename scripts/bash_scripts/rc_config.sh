#!/bin/bash

cd $(git rev-parse --show-toplevel)

if [ -d ".venv" ]; then
	# Install UV
	curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="/scratch/alpine/$USER/bin" sh

	# Make sure to cache in scratch
	export UV_CACHE_DIR="/scratch/alpine/$USER/.uv"
	export PATH=$PATH:"/scratch/alpine/$USER/bin"

	uv venv -p 3.9
fi

uv pip install wandb

source .venv/bin/activate
uv pip install -e ../overcooked_ai
uv pip install pip==22 setuptools==62 torch==2.5.1
pip install -e .

