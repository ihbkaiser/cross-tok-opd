"""Keep Ray evidence persistent while Unix IPC sockets stay on local storage."""
import json
from pathlib import Path
import tempfile


def install_local_socket_layout():
    from ray._private.node import Node
    if getattr(Node, "_kdflow_local_socket_layout", False):
        return
    original = Node._init_temp

    def init_temp(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if not hasattr(self, "_sockets_dir") or not hasattr(self, "_session_dir"):
            raise RuntimeError("Unsupported Ray temp layout; use the qualified Ray runtime")
        # Distributed/FUSE volumes need not support AF_UNIX bind. This directory
        # contains IPC endpoints only; logs/session metadata stay in _session_dir.
        self._sockets_dir = tempfile.mkdtemp(prefix="rs-", dir="/tmp")
        record = {"session": self._session_dir, "socket_directory": self._sockets_dir,
                  "policy": "persistent-session-local-ipc-v1"}
        (Path(self._session_dir) / "kdflow-socket-layout.json").write_text(json.dumps(record))

    Node._init_temp = init_temp
    Node._kdflow_local_socket_layout = True
