"""CPU-testable SGLang request and hidden-state transport contract."""

import pickle

import numpy as np
import zmq


def _handle_generate(engine, request, data_socket, request_queue, response_queue):
    """Handle a generate request: run inference and send hidden states via ZMQ."""
    kwargs = request["kwargs"]

    generate_kwargs = {
        "sampling_params": kwargs["sampling_params"],
        "return_hidden_states": kwargs.get("return_hidden_states", True),
    }
    if kwargs.get("input_ids") is not None:
        generate_kwargs["input_ids"] = kwargs["input_ids"]
    else:
        generate_kwargs["prompt"] = kwargs["prompt"]
    if kwargs.get("image_data") is not None:
        generate_kwargs["image_data"] = kwargs["image_data"]

    outputs = engine.generate(**generate_kwargs)

    if len(outputs) != len(kwargs["loss_masks"]):
        raise RuntimeError("teacher output count differs from requested trajectories")
    selected_hiddens = []
    for output, mask in zip(outputs, kwargs["loss_masks"]):
        hs_np = output["meta_info"]["hidden_states"][0]
        if kwargs.get("input_ids") is not None and hs_np.shape[0] != mask.shape[0]:
            raise RuntimeError("teacher hidden-state length differs from explicit token trajectory")
        hs_np = hs_np[:mask.shape[0]]  # legacy text path may have been truncated
        hs_np = hs_np[mask]
        if not hs_np.flags['C_CONTIGUOUS']:
            hs_np = np.ascontiguousarray(hs_np)

        selected_hiddens.append(hs_np)

    # Validate every trajectory before acknowledging success. Otherwise the
    # client can wait on ZMQ for states that the failed worker will never send.
    response_queue.put({"type": "generate", "success": True, "num_samples": len(outputs)})
    for hs_np in selected_hiddens:
        meta = pickle.dumps({"shape": hs_np.shape, "dtype": str(hs_np.dtype)})
        data_socket.send(meta, flags=zmq.SNDMORE)
        data_socket.send(hs_np, copy=False)
