from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import hashlib
import copy

app = FastAPI()

# In-memory store for session states
SESSIONS = {}

DAG_NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]

REQUIRED_INPUT_KEYS = [
    "generation", "checksum", "canonicalData", "prepareCode",
    "prepareConfig", "trainCode", "trainConfig", "runtime",
    "evaluateCode", "evaluateConfig", "schemaDigest", "publishConfig"
]

EVENT_FIELDS = {
    "eventId", "revision", "node", "attempt",
    "status", "key", "artifactDigest", "receiptId"
}


def is_pos_int(val):
    """Checks if a value is a positive integer (excluding boolean)."""
    return isinstance(val, int) and not isinstance(val, bool) and val > 0


def compute_sha256(data_list):
    """Computes lowercase SHA-256 over exact UTF-8 compact JSON arrays."""
    json_bytes = json.dumps(data_list, separators=(',', ':'), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest().lower()


def get_canonical_event_json(ev_dict):
    """Computes compact canonical JSON representation for event ID uniqueness checks."""
    return json.dumps(ev_dict, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


class SessionState:
    def __init__(self, session_id, revision, inputs):
        self.session_id = session_id
        self.revision = revision
        self.inputs = inputs
        self.seen_events = {}  # eventId -> canonical JSON string
        self.cache = {}        # (node, cacheKey) -> {"artifactDigest": ..., "eventId": ..., "receiptId": ...}
        self.reset_node_states()

    def reset_node_states(self):
        """Clears active attempt and terminal state across revisions."""
        self.node_states = {
            n: {
                "status": "none",
                "attempt": 0,
                "key": None,
                "bound_artifact": None,
                "bound_receipt": None,
                "accepted_events": []
            }
            for n in DAG_NODES
        }


def compute_node_inputs_and_key(node, inputs, session_cache):
    """Computes dependency inputs dictionary and parent-gated cacheKey."""
    if node == "verify_data":
        dep_dict = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"]
        }
        key = compute_sha256([inputs["generation"], inputs["checksum"]])
        dep_dict["cacheKey"] = key
        return dep_dict, key

    elif node == "prepare":
        _, p_key = compute_node_inputs_and_key("verify_data", inputs, session_cache)
        p_reusable = ("verify_data", p_key) in session_cache if p_key else False

        dep_dict = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"]
        }
        if not p_reusable:
            dep_dict["cacheKey"] = None
            return dep_dict, None

        key = compute_sha256([inputs["canonicalData"], inputs["prepareCode"], inputs["prepareConfig"]])
        dep_dict["cacheKey"] = key
        return dep_dict, key

    elif node == "train":
        _, p_key = compute_node_inputs_and_key("prepare", inputs, session_cache)
        p_reusable = ("prepare", p_key) in session_cache if p_key else False

        if not p_reusable:
            dep_dict = {
                "prepareArtifact": None,
                "trainCode": inputs["trainCode"],
                "trainConfig": inputs["trainConfig"],
                "runtime": inputs["runtime"],
                "cacheKey": None
            }
            return dep_dict, None

        p_artifact = session_cache[("prepare", p_key)]["artifactDigest"]
        dep_dict = {
            "prepareArtifact": p_artifact,
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"]
        }
        key = compute_sha256([p_artifact, inputs["trainCode"], inputs["trainConfig"], inputs["runtime"]])
        dep_dict["cacheKey"] = key
        return dep_dict, key

    elif node == "evaluate":
        _, p_key = compute_node_inputs_and_key("train", inputs, session_cache)
        p_reusable = ("train", p_key) in session_cache if p_key else False

        if not p_reusable:
            dep_dict = {
                "trainArtifact": None,
                "canonicalData": inputs["canonicalData"],
                "evaluateCode": inputs["evaluateCode"],
                "evaluateConfig": inputs["evaluateConfig"],
                "cacheKey": None
            }
            return dep_dict, None

        p_artifact = session_cache[("train", p_key)]["artifactDigest"]
        dep_dict = {
            "trainArtifact": p_artifact,
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"]
        }
        key = compute_sha256([p_artifact, inputs["canonicalData"], inputs["evaluateCode"], inputs["evaluateConfig"]])
        dep_dict["cacheKey"] = key
        return dep_dict, key

    elif node == "register":
        _, p_key = compute_node_inputs_and_key("evaluate", inputs, session_cache)
        p_reusable = ("evaluate", p_key) in session_cache if p_key else False

        if not p_reusable:
            dep_dict = {
                "evaluateArtifact": None,
                "schemaDigest": inputs["schemaDigest"],
                "cacheKey": None
            }
            return dep_dict, None

        p_artifact = session_cache[("evaluate", p_key)]["artifactDigest"]
        dep_dict = {
            "evaluateArtifact": p_artifact,
            "schemaDigest": inputs["schemaDigest"]
        }
        key = compute_sha256([p_artifact, inputs["schemaDigest"]])
        dep_dict["cacheKey"] = key
        return dep_dict, key

    elif node == "publish":
        _, p_key = compute_node_inputs_and_key("register", inputs, session_cache)
        p_reusable = ("register", p_key) in session_cache if p_key else False

        if not p_reusable:
            dep_dict = {
                "registerArtifact": None,
                "publishConfig": inputs["publishConfig"],
                "cacheKey": None
            }
            return dep_dict, None

        p_artifact = session_cache[("register", p_key)]["artifactDigest"]
        dep_dict = {
            "registerArtifact": p_artifact,
            "publishConfig": inputs["publishConfig"]
        }
        key = compute_sha256([p_artifact, inputs["publishConfig"]])
        dep_dict["cacheKey"] = key
        return dep_dict, key

    return {}, None


@app.post("/pipeline")
async def pipeline(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    # 1. Structural Request Validation
    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    session_id = body.get("session")
    revision = body.get("revision")
    inputs = body.get("inputs")
    events = body.get("events")

    if not isinstance(session_id, str) or len(session_id) == 0:
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    if not is_pos_int(revision):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    if not isinstance(inputs, dict):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    for k in REQUIRED_INPUT_KEYS:
        v = inputs.get(k)
        if not isinstance(v, str) or len(v) == 0:
            return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    if not isinstance(events, list):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    # 2. Event Structural Validation
    for ev in events:
        if not isinstance(ev, dict):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if set(ev.keys()) != EVENT_FIELDS:
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev.get("eventId"), str) or len(ev["eventId"]) == 0:
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not is_pos_int(ev.get("revision")):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev.get("node"), str):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not is_pos_int(ev.get("attempt")):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev.get("status"), str):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev.get("key"), str) or len(ev["key"]) == 0:
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if ev.get("artifactDigest") is not None and not isinstance(ev["artifactDigest"], str):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if ev.get("receiptId") is not None and not isinstance(ev["receiptId"], str):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)

    # 3. Session & Revision Management
    session = SESSIONS.get(session_id)
    if not session:
        session = SessionState(session_id, revision, inputs)
        SESSIONS[session_id] = session
    else:
        if revision < session.revision:
            return JSONResponse({"error": "REVISION_CONFLICT"}, status_code=409)
        elif revision == session.revision:
            if inputs != session.inputs:
                return JSONResponse({"error": "REVISION_CONFLICT"}, status_code=409)
        else:
            session.revision = revision
            session.inputs = inputs
            session.reset_node_states()

    # Deep copy for atomic batch processing
    work = copy.deepcopy(session)

    accepted_event_ids = []
    ignored_event_ids = []

    # 4. Batch Event Processing
    for ev in events:
        ev_id = ev["eventId"]
        ev_json = get_canonical_event_json(ev)

        # Global Replay / Event ID Conflict Check
        if ev_id in work.seen_events:
            if work.seen_events[ev_id] == ev_json:
                ignored_event_ids.append(ev_id)
                continue
            else:
                return JSONResponse({"error": "EVENT_ID_CONFLICT"}, status_code=409)

        # Basic Filter Rules (Ignore if mismatch)
        if ev["revision"] != work.revision:
            ignored_event_ids.append(ev_id)
            continue

        node = ev["node"]
        if node not in DAG_NODES:
            ignored_event_ids.append(ev_id)
            continue

        status = ev["status"]
        if status not in ["started", "succeeded", "retryable_failed", "terminal_failed"]:
            ignored_event_ids.append(ev_id)
            continue

        # Artifact Digest Validation
        art_digest = ev["artifactDigest"]
        if status == "succeeded":
            if not isinstance(art_digest, str) or len(art_digest) == 0:
                ignored_event_ids.append(ev_id)
                continue
        else:
            if art_digest is not None:
                ignored_event_ids.append(ev_id)
                continue

        # Receipt Validation
        rcpt = ev["receiptId"]
        if status == "succeeded" and node in ["register", "publish"]:
            expected_rcpt = f"receipt:{node}:{ev['key']}"
            if rcpt != expected_rcpt:
                ignored_event_ids.append(ev_id)
                continue
        else:
            if rcpt is not None:
                ignored_event_ids.append(ev_id)
                continue

        # Key / Parent Availability Check
        _, expected_key = compute_node_inputs_and_key(node, work.inputs, work.cache)
        if not expected_key or ev["key"] != expected_key:
            ignored_event_ids.append(ev_id)
            continue

        node_st = work.node_states[node]

        # Transition Checks on Cached/Succeeded State
        if (node, expected_key) in work.cache or node_st["status"] == "succeeded":
            rec_art = (
                work.cache[(node, expected_key)]["artifactDigest"]
                if (node, expected_key) in work.cache
                else node_st["bound_artifact"]
            )
            if status == "succeeded":
                if art_digest != rec_art:
                    return JSONResponse({"error": "EVIDENCE_CONFLICT"}, status_code=409)
                else:
                    return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)
            else:
                return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        c_status = node_st["status"]
        c_attempt = node_st["attempt"]
        attempt = ev["attempt"]

        # Non-cached State Transitions
        if c_status == "none":
            if status == "started" and attempt == 1:
                node_st["status"] = "started"
                node_st["attempt"] = 1
                node_st["key"] = expected_key
                node_st["accepted_events"].append(ev_id)
            else:
                ignored_event_ids.append(ev_id)
                continue

        elif c_status == "started":
            if attempt < c_attempt:
                ignored_event_ids.append(ev_id)
                continue
            elif attempt == c_attempt:
                if status in ["succeeded", "retryable_failed", "terminal_failed"]:
                    node_st["status"] = status
                    node_st["accepted_events"].append(ev_id)
                    if status == "succeeded":
                        node_st["bound_artifact"] = art_digest
                        node_st["bound_receipt"] = rcpt
                        work.cache[(node, expected_key)] = {
                            "artifactDigest": art_digest,
                            "eventId": ev_id,
                            "receiptId": rcpt
                        }
                else:
                    return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)
            else:
                return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        elif c_status == "retryable_failed":
            if attempt <= c_attempt:
                ignored_event_ids.append(ev_id)
                continue
            elif attempt == c_attempt + 1:
                if status == "started":
                    node_st["status"] = "started"
                    node_st["attempt"] = attempt
                    node_st["accepted_events"].append(ev_id)
                else:
                    return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)
            else:
                return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        elif c_status == "terminal_failed":
            return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        work.seen_events[ev_id] = ev_json
        accepted_event_ids.append(ev_id)

  # 5. Build Response DAG Node States
    response_nodes = []
    upstream_terminal_flag = False

    for node in DAG_NODES:
        dep_dict, key = compute_node_inputs_and_key(node, work.inputs, work.cache)

        if upstream_terminal_flag:
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_TERMINAL"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": []
            })
            continue

        if key is None:
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_PENDING"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": []
            })
            continue

        if (node, key) in work.cache:
            cache_info = work.cache[(node, key)]
            response_nodes.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": ["CACHE_HIT"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": [cache_info["eventId"]]
            })
            continue

        node_st = work.node_states[node]
        c_status = node_st["status"]
        c_events = node_st["accepted_events"]

        if c_status == "none":
            response_nodes.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": ["CACHE_MISS"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": []
            })

        elif c_status == "started":
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["RUNNING"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": c_events
            })

        elif c_status == "retryable_failed":
            response_nodes.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": ["RETRYABLE_FAILURE"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": c_events
            })

        elif c_status == "terminal_failed":
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["TERMINAL_FAILURE"],
                "dependencyDigests": dep_dict,
                "triggeringEventIds": c_events
            })
            upstream_terminal_flag = True

    # Commit updated state on success
    SESSIONS[session_id] = work

    return JSONResponse({
        "revision": work.revision,
        "acceptedEventIds": accepted_event_ids,
        "ignoredEventIds": ignored_event_ids,
        "nodes": response_nodes
    })
