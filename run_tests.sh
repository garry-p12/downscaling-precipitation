#!/usr/bin/env bash
# Run both test suites in separate processes (see pytest.ini for why).
set -e
PY=${PYTHON:-python}
echo "=== pipeline suite"
"$PY" -m pytest -q tests
echo "=== deep-learning suite"
"$PY" -m pytest -q tests/test_deep.py -p no:cacheprovider
