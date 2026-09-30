"""Optional per-job port allocation for colocated independent Ray jobs."""
import os

def configured_port(name, default=None):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer TCP port") from exc
    if not 1024 <= port <= 65535:
        raise ValueError(f"{name} must be between 1024 and 65535")
    return port
