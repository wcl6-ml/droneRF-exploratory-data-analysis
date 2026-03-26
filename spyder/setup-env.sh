#!/usr/bin/env bash
set -e

ENV_NAME=${1:-spyder-test} # allow override via argument
PYPROJECT_FILE="pyproject.toml"

PYTHON_VERSION=$(grep -Po '(?<=requires-python = ")[^"]+' $PYPROJECT_FILE | grep -Po '\d+.\d+')

if [ -z "$PYTHON_VERSION" ]; then
echo "❌ Could not determine Python version from pyproject.toml"
exit 1
fi

echo "➡️ Using Python $PYTHON_VERSION"
echo "➡️ Environment name: $ENV_NAME"

conda create -n $ENV_NAME python=$PYTHON_VERSION -y

eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

conda install -c conda-forge spyder -y

pip install -e .[dev]

echo "✅ Environment '$ENV_NAME' ready!"
