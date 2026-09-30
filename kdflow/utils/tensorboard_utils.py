import numbers
import os
import threading
from pathlib import Path
from typing import Any, Callable, Mapping, Optional


def _scalar_value(value: Any) -> Optional[float]:
    """Convert scalar-like metric values without accepting arrays or vectors."""
    if isinstance(value, numbers.Real):
        return float(value)

    numel = getattr(value, "numel", None)
    if callable(numel):
        try:
            if int(numel()) != 1:
                return None
        except (TypeError, ValueError, RuntimeError):
            return None

    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
            return None
        if isinstance(scalar, numbers.Real):
            return float(scalar)
    return None


class TensorBoardLogger:
    """Small, thread-safe scalar logger shared by all KDFlow trainers."""

    def __init__(
        self,
        log_dir: str,
        *,
        flush_secs: int = 10,
        writer_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        if flush_secs <= 0:
            raise ValueError("tensorboard_flush_secs must be positive")
        if writer_factory is None:
            from torch.utils.tensorboard import SummaryWriter

            writer_factory = SummaryWriter

        self.log_dir = os.fspath(Path(log_dir).expanduser().resolve())
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self._writer = writer_factory(log_dir=self.log_dir, flush_secs=flush_secs)
        self._lock = threading.Lock()
        self._closed = False

    def log(self, metrics: Mapping[str, Any], *, step: int) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot write to a closed TensorBoard logger")
            for name, value in metrics.items():
                scalar = _scalar_value(value)
                if scalar is not None:
                    self._writer.add_scalar(name, scalar, global_step=step)
            # Flush each explicit training/resource publication. This keeps
            # event files useful even when a remote job exits unexpectedly.
            self._writer.flush()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._writer.flush()
            self._writer.close()
            self._closed = True


def create_tensorboard_logger(args, *, default_log_dir: str) -> Optional[TensorBoardLogger]:
    if not args.log.use_tensorboard:
        return None
    log_dir = args.log.tensorboard_log_dir or default_log_dir
    return TensorBoardLogger(
        log_dir,
        flush_secs=args.log.tensorboard_flush_secs,
    )
