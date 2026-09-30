"""Replay an internal parity capture on its saved student; never trains."""
import argparse
import json
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("capture", type=Path)
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--layout", choices=("batch", "unpadded"), default="batch")
    args = p.parse_args()
    from transformers import AutoModelForCausalLM
    meta = json.loads((args.capture / "metadata.json").read_text())
    assert meta["checkpoint_complete"], "Capture has no complete student checkpoint"
    assert torch.cuda.device_count() == 1, "Expose only the diagnostic GPU"
    batch = torch.load(args.capture / "batch.pt", map_location="cpu", weights_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.capture / "student", torch_dtype=getattr(torch, args.dtype),
        attn_implementation="eager", local_files_only=True, trust_remote_code=False).cuda()
    model.train(meta["model_training"])
    ids, attn, mask = (batch[k] for k in ("stu_input_ids", "stu_attn_mask", "stu_loss_mask"))
    mask = mask.bool()
    behavior = batch["stu_behavior_log_probs"][mask]
    results = []
    with torch.no_grad():
        groups = [list(range(len(ids)))] if args.layout == "batch" else [[i] for i in range(len(ids))]
        for group in groups:
            x, a, m = ids[group], attn[group], mask[group]
            if args.layout == "unpadded":
                keep = a[0].bool()
                x, a, m = x[:, keep], a[:, keep], m[:, keep]
            positions = a.long().cumsum(-1) - 1
            positions.masked_fill_(a == 0, 1)
            logits = model(input_ids=x.cuda(), attention_mask=a.cuda(),
                           position_ids=positions.cuda(), use_cache=False).logits
            selected = logits[m.cuda()].float() / batch["temperature"]
            labels = x.roll(-1, 1)[m].cuda()
            results.append(selected.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1).cpu())
            del logits, selected
    actual = torch.cat(results)
    real = torch.isfinite(behavior)
    report = dict(dtype=args.dtype, layout=args.layout)
    for name, reference in (("behavior", behavior), ("captured_trainer", batch["actual_log_probs"])):
        gap = (actual[real] - reference[real]).abs().float()
        report[name] = dict(mean=float(gap.mean()), p99=float(torch.quantile(gap, .99)),
                           maximum=float(gap.max()), fraction_above_0p5=float((gap > .5).float().mean()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
