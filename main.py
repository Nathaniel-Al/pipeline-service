from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import json
import hashlib
import copy

app = FastAPI()

# Global session storage mapping: session_id -> SessionState
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


def compute_sha256(data_list):
    """Computes lowercase SHA-256 over UTF-8 compact JSON array."""
    json_str = json.dumps(data_list, separators=(',', ':'))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest().lower()


def get_canonical_event_json(event_dict):
    """Computes compact canonical JSON for event ID collision checks."""
    return json.dumps(event_dict, sort_keys=True, separators=(',', ':'))


class SessionState:
    def __init__(self, session_id, revision, inputs):
        self.session_id = session_id
        self.revision = revision
        self.inputs = inputs
        # eventId -> canonical JSON string
        self.seen_events = {}
        # (node, cacheKey) -> {"artifactDigest": ..., "eventId": ..., "receiptId": ...}
        self.cache = {}
        # node -> {"status": ..., "attempt": ..., "key": ..., "bound_artifact": ..., "bound_receipt": ..., "accepted_events": []}
        self.node_states = {n: {"status": "none", "attempt": 0, "key": None, "bound_artifact": None, "bound_receipt": None, "accepted_events": []} for n in DAG_NODES}

    def reset_non_cached_node_states(self):
        """When revision changes, clear active execution state for non-cached nodes."""
        for n in DAG_NODES:
            curr = self.node_states[n]
            k = curr.get("key")
            if k and (n, k) in self.cache:
                continue
            self.node_states[n] = {
                "status": "none",
                "attempt": 0,
                "key": None,
                "bound_artifact": None,
                "bound_receipt": None,
                "accepted_events": []
            }


def compute_node_inputs_and_key(node, inputs, session_cache):
    """Computes dependency inputs dictionary and cacheKey for a given DAG node."""
    if node == "verify_data":
        dep_inputs = {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"]
        }
        key_data = [inputs["generation"], inputs["checksum"]]
        return dep_inputs, compute_sha256(key_data)

    elif node == "prepare":
        parent_cache = session_cache.get(("verify_data", compute_sha256([inputs["generation"], inputs["checksum"]])))
        if not parent_cache:
            return {
                "canonicalData": inputs["canonicalData"],
                "prepareCode": inputs["prepareCode"],
                "prepareConfig": inputs["prepareConfig"],
                "cacheKey": None
            }, None

        dep_inputs = {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"]
        }
        key_data = [inputs["canonicalData"], inputs["prepareCode"], inputs["prepareConfig"]]
        return dep_inputs, compute_sha256(key_data)

    elif node == "train":
        p_inputs, p_key = compute_node_inputs_and_key("prepare", inputs, session_cache)
        parent_cache = session_cache.get(("prepare", p_key)) if p_key else None
        if not parent_cache:
            return {
                "prepareArtifact": None,
                "trainCode": inputs["trainCode"],
                "trainConfig": inputs["trainConfig"],
                "runtime": inputs["runtime"],
                "cacheKey": None
            }, None

        p_artifact = parent_cache["artifactDigest"]
        dep_inputs = {
            "prepareArtifact": p_artifact,
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"]
        }
        key_data = [p_artifact, inputs["trainCode"], inputs["trainConfig"], inputs["runtime"]]
        return dep_inputs, compute_sha256(key_data)

    elif node == "evaluate":
        p_inputs, p_key = compute_node_inputs_and_key("train", inputs, session_cache)
        parent_cache = session_cache.get(("train", p_key)) if p_key else None
        if not parent_cache:
            return {
                "trainArtifact": None,
                "canonicalData": inputs["canonicalData"],
                "evaluateCode": inputs["evaluateCode"],
                "evaluateConfig": inputs["evaluateConfig"],
                "cacheKey": None
            }, None

        p_artifact = parent_cache["artifactDigest"]
        dep_inputs = {
            "trainArtifact": p_artifact,
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"]
        }
        key_data = [p_artifact, inputs["canonicalData"], inputs["evaluateCode"], inputs["evaluateConfig"]]
        return dep_inputs, compute_sha256(key_data)

    elif node == "register":
        p_inputs, p_key = compute_node_inputs_and_key("evaluate", inputs, session_cache)
        parent_cache = session_cache.get(("evaluate", p_key)) if p_key else None
        if not parent_cache:
            return {
                "evaluateArtifact": None,
                "schemaDigest": inputs["schemaDigest"],
                "cacheKey": None
            }, None

        p_artifact = parent_cache["artifactDigest"]
        dep_inputs = {
            "evaluateArtifact": p_artifact,
            "schemaDigest": inputs["schemaDigest"]
        }
        key_data = [p_artifact, inputs["schemaDigest"]]
        return dep_inputs, compute_sha256(key_data)

    elif node == "publish":
        p_inputs, p_key = compute_node_inputs_and_key("register", inputs, session_cache)
        parent_cache = session_cache.get(("register", p_key)) if p_key else None
        if not parent_cache:
            return {
                "registerArtifact": None,
                "publishConfig": inputs["publishConfig"],
                "cacheKey": None
            }, None

        p_artifact = parent_cache["artifactDigest"]
        dep_inputs = {
            "registerArtifact": p_artifact,
            "publishConfig": inputs["publishConfig"]
        }
        key_data = [p_artifact, inputs["publishConfig"]]
        return dep_inputs, compute_sha256(key_data)

    return {}, None


@app.post("/pipeline")
async def pipeline(request: Request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    # 1. Request Structure Validation
    if not isinstance(body, dict):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    session_id = body.get("session")
    revision = body.get("revision")
    inputs = body.get("inputs")
    events = body.get("events")

    if not isinstance(session_id, str) or len(session_id) == 0:
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    if not isinstance(revision, int) or revision <= 0 or isinstance(revision, bool):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    if not isinstance(inputs, dict):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    for k in REQUIRED_INPUT_KEYS:
        val = inputs.get(k)
        if not isinstance(val, str) or len(val) == 0:
            return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    if not isinstance(events, list):
        return JSONResponse({"error": "INVALID_REQUEST"}, status_code=409)

    # 2. Event Structural Validation
    for ev in events:
        if not isinstance(ev, dict):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if set(ev.keys()) != EVENT_FIELDS:
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev["eventId"], str) or len(ev["eventId"]) == 0:
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev["revision"], int) or ev["revision"] <= 0 or isinstance(ev["revision"], bool):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev["node"], str):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev["attempt"], int) or ev["attempt"] <= 0 or isinstance(ev["attempt"], bool):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev["status"], str):
            return JSONResponse({"error": "INVALID_EVENT"}, status_code=409)
        if not isinstance(ev["key"], str) or len(ev["key"]) == 0:
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
            session.reset_non_cached_node_states()

    # Create working copy for atomic batch rollback on error
    work = copy.deepcopy(session)

    accepted_event_ids = []
    ignored_event_ids = []

    # 4. Event Batch Processing
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

        # Basic filtering: Revision, Node & Format validation
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

        # Artifact Digest validation
        art_digest = ev["artifactDigest"]
        if status == "succeeded":
            if not isinstance(art_digest, str) or len(art_digest) == 0:
                ignored_event_ids.append(ev_id)
                continue
        else:
            if art_digest is not None:
                ignored_event_ids.append(ev_id)
                continue

        # Receipt validation
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

        # Check key against expected node key based on inputs and parent reusable state
        _, expected_key = compute_node_inputs_and_key(node, work.inputs, work.cache)
        if not expected_key or ev["key"] != expected_key:
            ignored_event_ids.append(ev_id)
            continue

        # Retrieve current node state
        curr_state = work.node_states[node]
        c_status = curr_state["status"]
        c_attempt = curr_state["attempt"]
        c_key = curr_state["key"]

        # Check if already cached in session
        if (node, expected_key) in work.cache:
            bound_info = work.cache[(node, expected_key)]
            if status == "succeeded":
                if art_digest != bound_info["artifactDigest"]:
                    return JSONResponse({"error": "EVIDENCE_CONFLICT"}, status_code=409)
                else:
                    return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)
            else:
                return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        # State transition evaluation
        attempt = ev["attempt"]

        if c_status == "none":
            if status == "started" and attempt == 1:
                curr_state["status"] = "started"
                curr_state["attempt"] = 1
                curr_state["key"] = expected_key
                curr_state["accepted_events"].append(ev_id)
            else:
                ignored_event_ids.append(ev_id)
                continue

        elif c_status == "started":
            if attempt < c_attempt:
                ignored_event_ids.append(ev_id)
                continue
            elif attempt == c_attempt:
                if status in ["succeeded", "retryable_failed", "terminal_failed"]:
                    curr_state["status"] = status
                    curr_state["accepted_events"].append(ev_id)
                    if status == "succeeded":
                        curr_state["bound_artifact"] = art_digest
                        curr_state["bound_receipt"] = rcpt
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
            if attempt < c_attempt + 1:
                ignored_event_ids.append(ev_id)
                continue
            elif attempt == c_attempt + 1 and status == "started":
                curr_state["status"] = "started"
                curr_state["attempt"] = attempt
                curr_state["accepted_events"].append(ev_id)
            else:
                return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        elif c_status == "succeeded":
            if status == "succeeded" and art_digest != curr_state["bound_artifact"]:
                return JSONResponse({"error": "EVIDENCE_CONFLICT"}, status_code=409)
            else:
                return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        elif c_status == "terminal_failed":
            return JSONResponse({"error": "STATUS_CONFLICT"}, status_code=409)

        # Mark event accepted and store canonical representation
        work.seen_events[ev_id] = ev_json
        accepted_event_ids.append(ev_id)

    # 5. Build Response DAG Node States
    response_nodes = []
    upstream_terminal_flag = False

    for node in DAG_NODES:
        dep_inputs, key = compute_node_inputs_and_key(node, work.inputs, work.cache)
        dep_digests = copy.deepcopy(dep_inputs)
        dep_digests["cacheKey"] = key

        if upstream_terminal_flag:
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_TERMINAL"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": []
            })
            continue

        if key is None:
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["UPSTREAM_PENDING"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": []
            })
            continue

        # Check content-addressed cache
        if (node, key) in work.cache:
            cache_info = work.cache[(node, key)]
            response_nodes.append({
                "node": node,
                "action": "reuse",
                "reasonCodes": ["CACHE_HIT"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": [cache_info["eventId"]]
            })
            continue

        curr_state = work.node_states[node]
        c_status = curr_state["status"]
        c_events = curr_state["accepted_events"]

        if c_status == "none":
            response_nodes.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": ["CACHE_MISS"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": []
            })

        elif c_status == "started":
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["RUNNING"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": c_events
            })

        elif c_status == "retryable_failed":
            response_nodes.append({
                "node": node,
                "action": "rerun",
                "reasonCodes": ["RETRYABLE_FAILURE"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": c_events
            })

        elif c_status == "terminal_failed":
            response_nodes.append({
                "node": node,
                "action": "block",
                "reasonCodes": ["TERMINAL_FAILURE"],
                "dependencyDigests": dep_digests,
                "triggeringEventIds": c_events
            })
            upstream_terminal_flag = True

    # Commit successful working state copy back to session
    SESSIONS[session_id] = work

    return JSONResponse({
        "revision": work.revision,
        "acceptedEventIds": accepted_event_ids,
        "ignoredEventIds": ignored_event_ids,
        "nodes": response_nodes
    })
