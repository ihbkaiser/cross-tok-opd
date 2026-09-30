#!/bin/sh
set -eu
umask 077
exec /usr/bin/python3 -u "$(dirname "$0")/managed_target.py" "$@"
