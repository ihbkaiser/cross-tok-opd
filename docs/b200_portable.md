# B200 portable export

Private HF dataset: codemaivanngu/simct-b200-portable-cu130
Revision: 7b369827eba8c5b20405eec7270b67b17b500d90
Runtime archive: 7276682121 bytes. Source archive: 7413760 bytes.
Runtime image pinned in exporter; source archive from41eb5b3.
Export implementation ce4a6ce. Export app ap-iLLihpba7ok5YQvxiOf5Xn;
upload recovery app ap-X7kgdKwzdXZnBRLoqhPyUc (CPU only).
The first upload failed because image sets HF_HUB_OFFLINE; online retry reused
existing archive. Final upload receipt confirms private repo, sizes and LFS hashes.
Archive allowlist and bash syntax checks passed; CPU torch/transformers/sglang
imports passed before upload. GPU target smoke remains pending user execution.

Target supplied: Ubuntu24.04.3/glibc2.39/x86_64/B200/driver580.105.08.
Default CUDA_VISIBLE_DEVICES=0. GPU1 is occupied; no target training launched.
Download pinned revision, run bash install.sh as a user permitted to create
listed absolute runtime paths. Installer refuses any existing target path.
It verifies SHA256, extracts runtime and source separately, runs CUDA matmul
smoke; not a complete SGLang/training qualification. Needs about25GB space.
No weights/data/tokens/training output bundled. The source's existing Phi
launcher must not be mistaken for the future Qwen to Gemma training recipe.
