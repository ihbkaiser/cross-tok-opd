#!/bin/sh
set -eu
umask 077
exec /usr/bin/python3 -u /workspace/SimCT/experiments/environments/cpu_logits_attempt.py "$@"
