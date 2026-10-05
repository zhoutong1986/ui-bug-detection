#!/usr/bin/env bash
set -e

python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -c "print('Environment ready')"
