# Meta/energy asset unblock — 2026-09-15

Read-only inventory prepared from HEAD `319c1cb401b7d11a7fbbba631307f5369de5e203`.

- `inventory.json`: source-traced candidates, hashes, lineage, compatibility and permission status.
- `company_read_only_inventory.sh`: run on the company host with the exact campaign case. It reads campaign/config/receipt metadata, hashes exact candidate paths, checks the production meta contract and verifies the trained energy checkpoint. It does not copy, upload, submit, restart or mutate queue state.

Current decision: **company-only P1 route**. Modal remains blocked because the allowed volume does not contain the exact `selected.parquet` and trained `energy-select-4.pt`. Historical paths are candidates until receipts and owner permission establish identity and use rights.
