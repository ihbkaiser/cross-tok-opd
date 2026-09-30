# External eval telemetry

Run eval_telemetry.py --plan /path/plan.json --out /new/file.jsonl on the company
node. Defaults to ten minutes sampled every five seconds, bounded to one hour.
It launches no GPU job and changes no queue files, source hashes or checkpoints.
Only the new output file is written, using exclusive creation.

Records GPU utilization/power/VRAM, per-cell response and score row counts,
response/score rates, backlog, last server decode batch/throughput timestamp,
worker descendant CPU ticks/RSS, host RAM/load/pressure and visible cgroup limits.
It does not record answer text, test payloads, secrets or process command lines.
Journal counting is incremental after the initial scan. Rows can race live writes;
use the queue's own verified metrics for scientific scores. Last decode metrics
can be stale; inspect timestamps. Summed process RSS double-counts shared memory;
short-lived scorers may disappear between samples. Visible cgroup limits do not
establish parent limits or exclusive allocation. Dataset/response dictionaries
remain resident in the eval worker independently of the telemetry collector.

Interpretation: sustained backlog growth with low running requests indicates
backpressure; low requests with no backlog needs further client/I/O inspection.
Scorer limits may constrain throughput even if host CPU appears idle. A GPU-util
spike alone cannot identify prefill or prove throughput improvement.
