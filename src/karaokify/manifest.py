"""JSON result documents (the --json contract) and the JSONL --manifest audit trail."""

from __future__ import annotations

import datetime
import json
import sys
import threading
from pathlib import Path

from . import __version__

_manifest_lock = threading.Lock()

ACTIONS = ("processed", "skipped", "failed", "planned")


def make_result(
    input_path,
    output,
    action: str,
    *,
    model=None,
    device=None,
    stems=None,
    timings=None,
    input_duration_s=None,
    output_size_bytes=None,
    reason=None,
) -> dict:
    assert action in ACTIONS
    result = {
        "input": str(input_path),
        "output": str(output) if output is not None else None,
        "action": action,
        "model": model,
        "device": device,
        "stems": stems,  # {stem_name: gain_db | null}; null = dropped (-inf)
        "timings": timings,  # {probe_s, separate_s, remux_s, total_s} | null
        "input_duration_s": input_duration_s,
        "output_size_bytes": output_size_bytes,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def make_document(results: list[dict]) -> dict:
    summary = {action: 0 for action in ACTIONS}
    for result in results:
        summary[result["action"]] += 1
    return {"version": __version__, "results": results, "summary": summary}


def emit(document: dict, stream=None) -> None:
    """Write exactly one JSON document to stdout (the --json contract)."""
    stream = stream or sys.stdout
    json.dump(document, stream, indent=2, allow_nan=False)
    stream.write("\n")
    stream.flush()


def append_manifest(path: Path, result: dict) -> None:
    record = {
        "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **result,
    }
    line = json.dumps(record, allow_nan=False)
    with _manifest_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def schema_path() -> Path:
    """Path of the checked-in JSON schema for the --json result document."""
    return Path(__file__).with_name("schema.json")
