"""One-off: upload the existing qa_comments.json into the Supabase reviews table.

Usage: SUPABASE_URL=https://xxx.supabase.co SUPABASE_KEY=<anon key> python migrate_comments.py
"""
import json
import os
from pathlib import Path

import requests

url = os.environ["SUPABASE_URL"].rstrip("/")
key = os.environ["SUPABASE_KEY"]
comments = json.loads((Path(__file__).parent / "qa_comments.json").read_text(encoding="utf-8"))

rows = []
for filename, models in comments.items():
    for model_key, items in models.items():
        if not isinstance(items, dict):
            continue
        for qa_index, data in items.items():
            rows.append(
                {"filename": filename, "model_key": model_key, "qa_index": int(qa_index), "data": data}
            )

resp = requests.post(
    f"{url}/rest/v1/reviews",
    params={"on_conflict": "filename,model_key,qa_index"},
    headers={
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    },
    json=rows,
    timeout=60,
)
resp.raise_for_status()
print(f"Uploaded {len(rows)} reviews")
