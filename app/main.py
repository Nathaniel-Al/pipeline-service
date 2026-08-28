from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
DB_PATH = os.environ.get("QUANTIZE_DB", "/tmp/quantize_state.sqlite3")
DB_LOCK = threading.RLock()

FREEZE_CODES = {
    "INVALID_INPUT",
    "UNALLOWED_UNSUPPORTED_REASON",
    "NOT_LOADABLE",
    "CALIBRATION_MISMATCH",
    "TOKENIZER_MISMATCH",
}


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.execute(
        "CREATE TABLE IF NOT EXISTS freezes ("
        "freeze_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, response_json TEXT NOT NULL)"
    )
    c.commit()
    return c


DB = conn()


def invalid_input() -> JSONResponse:
    return JSONResponse({"error": "INVALID_INPUT"}, status_code=400)


def conflict() -> JSONResponse:
    return JSONResponse({"error": "FREEZE_ID_CONFLICT"}, status_code=409)


def utf8(s: str) -> bytes:
    return s.encode("utf-8", "strict")


def compact_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8", "strict")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code_list(codes: set[str] | list[str]) -> list[str]:
    return sorted(set(codes), key=utf8)


def nonempty_string(x: Any) -> bool:
    return isinstance(x, str) and len(x) > 0


def unique_strings(x: Any) -> bool:
    return (
        isinstance(x, list)
        and all(nonempty_string(v) for v in x)
        and len(x) == len(set(x))
    )


def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def safe_nonnegative_integer(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 9007199254740991


def binary(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and (x == 0 or x == 1)


def round12(x: float) -> float:
    return round(x, 12)


def validate_files(files: Any) -> tuple[bool, list[dict[str, Any]], int | None, str | None]:
    if not isinstance(files, dict) or len(files) == 0:
        return False, [], None, None

    inventory: list[dict[str, Any]] = []
    for filename, text in files.items():
        if not isinstance(filename, str) or not isinstance(text, str):
            return False, [], None, None
        try:
            raw = text.encode("utf-8", "strict")
        except UnicodeEncodeError:
            return False, [], None, None
        inventory.append(
            {
                "name": filename,
                "bytes": len(raw),
                "sha256": sha256(raw),
            }
        )

    inventory.sort(key=lambda x: utf8(x["name"]))
    total = sum(x["bytes"] for x in inventory)
    package = sha256(compact_json(inventory))
    return True, inventory, total, package


def freeze_valid(body: Any) -> bool:
    if not isinstance(body, dict) or body.get("phase") != "freeze":
        return False
    if not isinstance(body.get("freezeId"), str) or not (1 <= len(body["freezeId"]) <= 128):
        return False
    if not nonempty_string(body.get("calibrationDigest")):
        return False
    if not nonempty_string(body.get("tokenizerDigest")):
        return False
    if not unique_strings(body.get("allowedUnsupportedReasons")):
        return False

    candidates = body.get("candidates")
    if not isinstance(candidates, list) or len(candidates) == 0:
        return False

    names: list[str] = []
    for c in candidates:
        if not isinstance(c, dict):
            return False
        if not nonempty_string(c.get("name")):
            return False
        names.append(c["name"])
        if "files" not in c or not isinstance(c["files"], dict):
            # This is candidate-level invalid file data, not a whole-request
            # shape error. build_freeze() will emit empty manifest fields.
            continue
        if "loadable" not in c or not isinstance(c["loadable"], bool):
            return False
        if not nonempty_string(c.get("calibrationDigest")):
            return False
        if not nonempty_string(c.get("tokenizerDigest")):
            return False
        if "unsupportedReason" in c and not nonempty_string(c["unsupportedReason"]):
            return False

    return len(names) == len(set(names))


def build_freeze(body: dict[str, Any]) -> dict[str, Any]:
    allowed = set(body["allowedUnsupportedReasons"])
    result: list[dict[str, Any]] = []

    for c in body["candidates"]:
        file_ok, inventory, total, package = validate_files(c.get("files"))
        reasons: set[str] = set()
        unsupported_reason = c.get("unsupportedReason")

        if not file_ok:
            reasons.add("INVALID_INPUT")
            status = "invalid"
            inventory = []
            total = None
            package = None
        elif unsupported_reason is not None and unsupported_reason in allowed:
            status = "unsupported"
        else:
            status = "frozen"
            if unsupported_reason is not None:
                reasons.add("UNALLOWED_UNSUPPORTED_REASON")
            if not c.get("loadable", False):
                reasons.add("NOT_LOADABLE")
            if c.get("calibrationDigest") != body["calibrationDigest"]:
                reasons.add("CALIBRATION_MISMATCH")
            if c.get("tokenizerDigest") != body["tokenizerDigest"]:
                reasons.add("TOKENIZER_MISMATCH")
            if reasons:
                status = "invalid"

        result.append(
            {
                "name": c["name"],
                "status": status,
                "inventory": inventory,
                "totalBytes": total,
                "packageDigest": package,
                "reasonCodes": code_list(reasons),
            }
        )

    result.sort(key=lambda x: utf8(x["name"]))
    return {"freezeId": body["freezeId"], "candidates": result}


def stored(freeze_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
    row = DB.execute(
        "SELECT request_json, response_json FROM freezes WHERE freeze_id = ?",
        (freeze_id,),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0]), json.loads(row[1])


def select_shape(body: Any) -> bool:
    return (
        isinstance(body, dict)
        and body.get("phase") == "select"
        and isinstance(body.get("freezeId"), str)
        and len(body["freezeId"]) > 0
        and isinstance(body.get("candidates"), list)
        and isinstance(body.get("rows"), list)
        and isinstance(body.get("policy"), dict)
    )


def validate_policy(policy: Any) -> bool:
    if not isinstance(policy, dict):
        return False
    required = ("maxBytes", "aggregateFloor", "requiredSlices", "maxLatencyMs", "candidateOrder")
    if any(k not in policy for k in required):
        return False
    if not safe_nonnegative_integer(policy["maxBytes"]):
        return False
    if not finite_number(policy["aggregateFloor"]) or not 0 <= float(policy["aggregateFloor"]) <= 1:
        return False
    if not finite_number(policy["maxLatencyMs"]) or float(policy["maxLatencyMs"]) < 0:
        return False
    if not isinstance(policy["requiredSlices"], dict):
        return False
    for name, floor in policy["requiredSlices"].items():
        if not nonempty_string(name):
            return False
        if not finite_number(floor) or not 0 <= float(floor) <= 1:
            return False
    return unique_strings(policy["candidateOrder"])


def manifest_valid(c: Any) -> tuple[bool, int | None, str | None]:
    if not isinstance(c, dict) or not isinstance(c.get("inventory"), list):
        return False, None, None

    inv = c["inventory"]
    names: list[str] = []
    normalized: list[dict[str, Any]] = []
    for item in inv:
        if not isinstance(item, dict) or list(item.keys()) != ["name", "bytes", "sha256"]:
            return False, None, None
        if not nonempty_string(item["name"]):
            return False, None, None
        if not safe_nonnegative_integer(item["bytes"]):
            return False, None, None
        digest = item["sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            return False, None, None
        names.append(item["name"])
        normalized.append(
            {"name": item["name"], "bytes": item["bytes"], "sha256": digest}
        )

    if len(names) != len(set(names)):
        return False, None, None
    if normalized != sorted(normalized, key=lambda x: utf8(x["name"])):
        return False, None, None

    total = sum(x["bytes"] for x in normalized)
    digest = sha256(compact_json(normalized))
    if c.get("totalBytes") != total or c.get("packageDigest") != digest:
        return False, None, None
    return True, total, digest


def latency(latencies: Any, name: str) -> tuple[bool, float | int | None]:
    if not isinstance(latencies, dict) or name not in latencies:
        return False, None
    value = latencies[name]
    if not finite_number(value) or float(value) < 0:
        return False, None
    return True, value


def select(body: dict[str, Any], stored_response: dict[str, Any]) -> dict[str, Any]:
    stored_candidates = stored_response.get("candidates", [])
    submitted = body["candidates"]
    policy = body["policy"]
    rows = body["rows"]
    latencies = body.get("latencies")

    global_codes: set[str] = set()
    policy_ok = validate_policy(policy)
    if not policy_ok:
        global_codes.add("INVALID_POLICY")

    stored_names = [c.get("name") for c in stored_candidates if isinstance(c, dict)]
    submitted_names = [c.get("name") if isinstance(c, dict) else None for c in submitted]
    order = policy.get("candidateOrder") if isinstance(policy, dict) else []

    if (
        not unique_strings(order)
        or set(order) != set(stored_names)
        or set(order) != set(x for x in submitted_names if isinstance(x, str))
        or len(submitted_names) != len(stored_names)
        or len([x for x in submitted_names if isinstance(x, str)]) != len(submitted_names)
    ):
        global_codes.add("INVALID_POLICY")

    stored_by_name = {c["name"]: c for c in stored_candidates if isinstance(c, dict) and "name" in c}
    submitted_by_name = {c["name"]: c for c in submitted if isinstance(c, dict) and "name" in c}

    # Results use candidateOrder. UTF-8 name is the deterministic fallback.
    names = list(stored_by_name.keys())
    names.sort(key=lambda n: (order.index(n) if n in order else len(order), utf8(n)))

    results: list[dict[str, Any]] = []
    required_slices = policy.get("requiredSlices", {}) if policy_ok else {}

    for name in names:
        c = stored_by_name[name]
        reasons = set(c.get("reasonCodes", []))

        if submitted_by_name.get(name) != c:
            reasons.add("INVALID_LINEAGE")

        manifest_ok, total_bytes, _ = manifest_valid(c)
        if not manifest_ok:
            reasons.add("INVALID_MANIFEST")
            total_bytes = None

        if c.get("status") != "frozen":
            reasons.add("NOT_FROZEN")

        aggregate = None
        slices: dict[str, Any] = {}
        predictions_ok = True
        values: list[tuple[str, Any, Any]] = []

        if not isinstance(rows, list):
            predictions_ok = False
        else:
            for row in rows:
                if not isinstance(row, dict):
                    predictions_ok = False
                    break
                if "label" not in row or "slice" not in row or "predictions" not in row:
                    predictions_ok = False
                    break
                if not binary(row["label"]) or not isinstance(row["slice"], str) or not isinstance(row["predictions"], dict):
                    predictions_ok = False
                    break
                if name not in row["predictions"] or not binary(row["predictions"][name]):
                    predictions_ok = False
                    break
                values.append((row["slice"], row["label"], row["predictions"][name]))

        if not predictions_ok:
            reasons.add("INVALID_PREDICTIONS")
        elif values:
            aggregate = round12(sum(1 for _, y, p in values if y == p) / len(values))
            for slice_name in sorted(required_slices.keys(), key=utf8):
                subset = [(y, p) for s, y, p in values if s == slice_name]
                if not subset:
                    slices[slice_name] = None
                    reasons.add(f"MISSING_SLICE:{slice_name}")
                else:
                    acc = round12(sum(1 for y, p in subset if y == p) / len(subset))
                    slices[slice_name] = acc
                    if acc < float(required_slices[slice_name]):
                        reasons.add(f"SLICE_FLOOR:{slice_name}")
        else:
            # Empty rows cannot satisfy an aggregate floor.
            reasons.add("AGGREGATE_FLOOR")
            for slice_name in sorted(required_slices.keys(), key=utf8):
                slices[slice_name] = None
                reasons.add(f"MISSING_SLICE:{slice_name}")

        if policy_ok:
            if aggregate is None or aggregate < float(policy["aggregateFloor"]):
                reasons.add("AGGREGATE_FLOOR")

        latency_ok, latency_value = latency(latencies, name)
        if not latency_ok:
            reasons.add("LATENCY_LIMIT")
            latency_value = None
        elif policy_ok and float(latency_value) > float(policy["maxLatencyMs"]):
            reasons.add("LATENCY_LIMIT")

        if policy_ok:
            if total_bytes is None or total_bytes > policy["maxBytes"]:
                reasons.add("SIZE_LIMIT")

        if global_codes:
            reasons.update(global_codes)

        results.append(
            {
                "name": name,
                "aggregate": aggregate,
                "slices": slices,
                "totalBytes": total_bytes,
                "latencyMs": latency_value,
                "admitted": len(reasons) == 0,
                "reasonCodes": code_list(reasons),
            }
        )

    admitted = [r for r in results if r["admitted"]]
    winner = None
    if admitted:
        positions = {name: i for i, name in enumerate(order)}
        winner = min(
            admitted,
            key=lambda r: (
                r["totalBytes"],
                float(r["latencyMs"]),
                positions.get(r["name"], len(order)),
                utf8(r["name"]),
            ),
        )

    return {
        "freezeId": body["freezeId"],
        "selected": winner["name"] if winner else None,
        "results": results,
        "packageManifest": stored_by_name[winner["name"]] if winner else None,
    }


@app.post("/quantize")
async def quantize(request: Request):
    try:
        raw = await request.body()
        try:
            body = json.loads(
                raw.decode("utf-8", "strict"),
                object_pairs_hook=lambda pairs: duplicate_check_object(pairs),
            )
        except (UnicodeDecodeError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
            return invalid_input()
    except Exception:
        return invalid_input()

    if not isinstance(body, dict):
        return invalid_input()

    phase = body.get("phase")
    if phase == "freeze":
        if not freeze_valid(body):
            return invalid_input()
        freeze_id = body["freezeId"]
        with DB_LOCK:
            existing = stored(freeze_id)
            if existing is not None:
                old_request, old_response = existing
                if old_request == body:
                    return JSONResponse(old_response)
                return conflict()

            response = build_freeze(body)
            DB.execute(
                "INSERT INTO freezes(freeze_id, request_json, response_json) VALUES (?, ?, ?)",
                (
                    freeze_id,
                    json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                    json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
                ),
            )
            DB.commit()
            return JSONResponse(response)

    if phase == "select":
        if not select_shape(body):
            return invalid_input()
        with DB_LOCK:
            existing = stored(body["freezeId"])
        if existing is None:
            stored_response = {"freezeId": body["freezeId"], "candidates": []}
        else:
            _, stored_response = existing
        return JSONResponse(select(body, stored_response))

    return invalid_input()


def duplicate_check_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON object key")
        out[key] = value
    return out


@app.get("/health")
def health():
    return {"status": "ok"}
