"""Opt-in, internal-only evidence for failed single-GPU behavior parity."""
import json
import os
from pathlib import Path
import tempfile

import torch


def capture_failure(algorithm, batch, logits, labels, behavior, temperature, error):
    root = os.environ.get("MP_PARITY_CAPTURE_DIR")
    if not root:
        return
    # Saving a sharded model from only the failing rank can deadlock peers.
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise RuntimeError("Parity capture requires exactly one distributed rank")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = Path(tempfile.mkdtemp(prefix="failure-", dir=root))
    mask = batch["stu_loss_mask"].bool()
    actual = (logits.detach().float() / float(temperature)).log_softmax(-1).gather(
        -1, labels.unsqueeze(-1)).squeeze(-1)
    positions = mask.nonzero()
    real = torch.isfinite(behavior)
    delta = (actual - behavior).abs()
    per_sample = []
    for sample in range(mask.shape[0]):
        selected = real & (positions[:, 0] == sample)
        values = delta[selected].float()
        per_sample.append(dict(sample=sample, tokens=int(selected.sum()),
            length=int(batch["stu_attn_mask"][sample].sum()),
            mean=float(values.mean()) if values.numel() else None,
            p99=float(torch.quantile(values, .99)) if values.numel() else None,
            maximum=float(values.max()) if values.numel() else None))
    # No prompts, credentials, teacher hidden states or vocabulary-sized logits.
    # Token IDs are still internal dataset content: never upload this artifact.
    evidence = {key: batch[key].detach().cpu() for key in
                ("stu_input_ids", "stu_attn_mask", "stu_loss_mask", "stu_behavior_log_probs")}
    evidence.update(actual_log_probs=actual.cpu(), positions=positions.cpu(),
                    temperature=float(temperature))
    torch.save(evidence, target / "batch.pt")
    metadata = dict(error=str(error), per_sample=per_sample,
        source_commit=os.environ.get("MP_SOURCE_COMMIT"),
        model_training=algorithm.student.training,
        attention_backend=algorithm.args.model.attn_implementation,
        torch_version=str(torch.__version__), checkpoint_complete=False)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    algorithm.strategy.save_model(algorithm.student, algorithm.student_tokenizer,
                                  str(target / "student"))
    metadata["checkpoint_complete"] = True
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"PARITY_CAPTURE={target}", flush=True)
