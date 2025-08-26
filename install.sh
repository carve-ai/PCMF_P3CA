#!/bin/bash
set -euo pipefail

# ---- PCMF_P3CA Installer ----
# Cross-platform (Linux/macOS) conda/mamba env setup with MOSEK license check
# Optionally builds Cython extension (admm_utils)

ENV_NAME="pcmf_p3ca_env"
PYTHON_VERSION="3.8"

echo "=== PCMF_P3CA Environment Installer ==="
echo "Target environment: $ENV_NAME (Python $PYTHON_VERSION)"
echo ""

# --- Step 1: Detect conda/mamba ---
if command -v mamba &> /dev/null; then
    CMD=mamba
    echo "Using mamba for fast environment creation."
elif command -v conda &> /dev/null; then
    CMD=conda
    echo "Using conda."
else
    echo "Error: conda or mamba not found. Please install Miniconda or Mambaforge first."
    exit 1
fi

# --- Step 2: Create environment ---
$CMD create -y -n "$ENV_NAME" python="$PYTHON_VERSION"

# --- Step 3: Activate environment (load shell hook then activate) ---
echo ""
echo "Activating environment..."
eval "$($CMD shell hook --shell bash)"
# if [ -n "${ZSH_VERSION:-}" ]; then
#   eval "$(conda shell.zsh hook)"
#   # eval "$($CMD shell.zsh hook)"mamba shell hook --shell bash
# else
#   eval "$(conda shell.bash hook)"
#   # eval "$($CMD shell.bash hook)"
# fi
$CMD activate "$ENV_NAME"

# --- Step 4: Install dependencies (Conda/conda-forge) ---
echo ""
echo "Installing core dependencies via $CMD..."
# temporarily set +eu to guard against unset variables on Mac when mamba/conda tries to deactivate/activate environment
set +eu
# =3.0.11
$CMD install -y \
  conda-forge::compilers \
  conda-forge::cvxpy=1.5.3 \
  cython \
  jupyter \
  matplotlib=3.5.2 \
  numba=0.56.4 \
  numpy=1.23.4 \
  pandas=1.3.4 \
  pip \
  conda-forge::scikit-learn=1.2.0 \
  conda-forge::scikit-sparse=0.4.14 \
  scipy=1.9.3 \
  seaborn=0.13.2 \
  tqdm=4.67.1

  # conda-forge::palmerpenguins \

$CMD deactivate
$CMD activate "$ENV_NAME"
# Change back to set -eu after install and activating environment
set -eu

# --- Step 5: Install pip dependencies (MOSEK, palmerpenguins) ---
echo ""
echo "Installing MOSEK via pip..."
pip install mosek palmerpenguins

# --- Step 6: Check for MOSEK license ---
echo ""
# /home/YOUR_USER_NAME/mosek/mosek.lic (Linux)
# /Users/YOUR_USER_NAME/mosek/mosek.lic (OSX)
# C:\Users\YOUR_USER_NAME\mosek\mosek.lic (Windows)
MOSEK_LIC="$HOME/mosek/mosek.lic"
if [ ! -f "$MOSEK_LIC" ]; then
    echo "=============================================================="
    echo " ERROR: MOSEK license file not found at $MOSEK_LIC"
    echo "        You must obtain a MOSEK license to use optimization features."
    echo ""
    echo "  1. Register for a free academic MOSEK license at:"
    echo "     https://www.mosek.com/license/request/?i=acp"
    echo ""
    echo "  2. Place your license file at: $MOSEK_LIC"
    echo ""
    echo " After obtaining and placing your license, rerun this script."
    echo "=============================================================="
    read -p "Press [Enter] to continue anyway, or Ctrl+C to abort..."
else
    echo "MOSEK license file found: $MOSEK_LIC"
fi

# --- Step 7: Build optional Cython extensions (ADMM utils) ---
echo ""
echo "Trying to build Cython extension admm_utils (optional, for fast ADMM, though numba is often faster)..."
if [ -f pcmf_p3ca/utils/setup_admm_utils.py ]; then
    set +e
    python pcmf_p3ca/utils/setup_admm_utils.py build_ext --inplace
    if [ $? -eq 0 ]; then
        echo "admm_utils compiled successfully."
    else
        echo "WARNING: Failed to build admm_utils. Using Python/Numba fallbacks."
    fi
    set -e
else
    echo "setup_admm_utils.py not found, skipping ADMM utils build."
fi

# --- Step 8: Install package editable ---
echo ""
echo "Installing the PCMF codebase in editable (developer) mode..."
pip install -e . --use-pep517

echo ""
echo "=== DONE ==="
echo "To start using, run:"
echo "    $CMD activate $ENV_NAME"
