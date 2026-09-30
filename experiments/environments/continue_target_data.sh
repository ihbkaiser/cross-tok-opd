#!/bin/sh
set -eu
umask 077
exec /usr/bin/python3 -u /workspace/SimCT/experiments/environments/continue_target_data.py "$@"
