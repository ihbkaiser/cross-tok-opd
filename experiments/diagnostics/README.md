# Parity diagnostics

parity_probe_v2.py - long-context logprob parity probe for the on-policy engine, run on the pod:
the trainer reference is ONE forward pass (never chunked), it gates on a self-check first, and it
measures the generated-token path at both single-request and training batch width.

parity_sweep_modal_20260930.py - the same measurement on a clean Modal B200 across engine
configurations (kv-cache dtype, radix cache). It is the probe that exposed a 2048-token "boundary"
as an artefact of a chunked trainer reference in an earlier version, which is why v2 refuses to
report anything until its self-check passes.
