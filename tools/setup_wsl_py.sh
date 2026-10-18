#!/bin/bash
# One-time WSL setup for our own code: Rust stable (core crate) and a Python venv with maturin,
# numpy and PyTorch (CUDA). Training runs in WSL because the Rust module builds as a Linux .so.
set -e
source ~/.cargo/env
rustup toolchain install stable --profile minimal
rustup component add --toolchain stable clippy rustfmt

python3 -m venv ~/venv/ic 2>/dev/null || {
  # Ubuntu ships python3 without ensurepip; bootstrap pip into the venv by hand.
  python3 -m venv --without-pip ~/venv/ic
  curl -sSfL https://bootstrap.pypa.io/get-pip.py | ~/venv/ic/bin/python
}
~/venv/ic/bin/pip install --upgrade pip
~/venv/ic/bin/pip install maturin numpy
~/venv/ic/bin/pip install torch --index-url https://download.pytorch.org/whl/cu130
~/venv/ic/bin/python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
