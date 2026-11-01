#!/bin/bash
set -euo pipefail
exec "$(dirname "$0")/../run_test.sh" "$@"
