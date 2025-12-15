#!/usr/bin/env bash
set -euo pipefail

if [[ "${SUPEREIGHT_DEBUG:-0}" == "1" ]]; then
  set -x
fi

# Print resource snapshot before running the workload
python /workspace/scripts/resource_report.py || true

echo "-> exec \"$@\""
exec "$@"
