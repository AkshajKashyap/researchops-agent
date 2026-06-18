import hashlib
import json


def make_workflow_id(query: str, corpus_index_dir: str) -> str:
    payload = {
        "query": query,
        "corpus_index_dir": corpus_index_dir,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"workflow_{digest[:12]}"
