import pytest

from kdflow.utils.tensorboard_utils import TensorBoardLogger, _scalar_value


class _RecordingWriter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, name, value, global_step):
        self.scalars.append((name, value, global_step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


def test_scalar_value_rejects_vector_metrics():
    class Vector:
        def numel(self):
            return 2

        def item(self):
            raise AssertionError("vector item() must not be called")

    assert _scalar_value(3) == 3.0
    assert _scalar_value(2.5) == 2.5
    assert _scalar_value(Vector()) is None


def test_tensorboard_logger_filters_and_flushes_scalars(tmp_path):
    writer = None

    def writer_factory(**kwargs):
        nonlocal writer
        writer = _RecordingWriter(**kwargs)
        return writer

    logger = TensorBoardLogger(
        str(tmp_path), flush_secs=1, writer_factory=writer_factory
    )
    logger.log(
        {
            "train/loss": 1.25,
            "train/global_step": 7,
            "metadata/non_scalar": "ignored",
        },
        step=7,
    )
    logger.close()
    logger.close()

    assert writer is not None
    assert writer.kwargs["flush_secs"] == 1
    assert writer.scalars == [
        ("train/loss", 1.25, 7),
        ("train/global_step", 7.0, 7),
    ]
    assert writer.flush_count == 2
    assert writer.close_count == 1


def test_tensorboard_logger_writes_readable_event_file(tmp_path):
    pytest.importorskip("tensorboard")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    logger = TensorBoardLogger(str(tmp_path), flush_secs=1)
    logger.log({"train/loss": 1.25, "train/global_step": 7}, step=7)
    logger.close()
    events = EventAccumulator(str(tmp_path))
    events.Reload()
    assert set(events.Tags()["scalars"]) == {"train/global_step", "train/loss"}
    loss = events.Scalars("train/loss")
    assert len(loss) == 1
    assert loss[0].step == 7
    assert loss[0].value == 1.25
