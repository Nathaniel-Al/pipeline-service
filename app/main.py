import hashlib
import json
import os
import sqlite3
import threading
from copy import deepcopy
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()
DB_PATH = os.environ.get("PIPELINE_DB", "/data/pipeline.db")
DB_DIR = os.path.dirname(DB_PATH) or "."
os.makedirs(DB_DIR, exist_ok=True)
LOCK = threading.RLock()

DAG = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENT = {
    "verify_data": None,
    "prepare": "verify_data",
    "train": "prepare",
    "evaluate": "train",
    "register": "evaluate",
    "publish": "register",
}

# The order here is significant: it is the exact order required for each
# content-addressed SHA-256 key.
NODE_DEPS = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig", "prepareArtifact"],
    "train": ["prepareArtifact", "trainCode", "trainConfig", "runtime"],
    "evaluate": ["trainArtifact", "canonicalData", "evaluateCode", "evaluateConfig"],
    "register": ["evaluateArtifact", "schemaDigest"],
    "publish": ["registerArtifact", "publishConfig"],
}

INPUT_KEYS = [
    "generation", "checksum", "canonicalData", "prepareCode", "prepareConfig",
    "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig",
    "schemaDigest", "publishConfig",
]
EVENT_FIELDS = [
    "eventId", "revision", "node", "attempt", "status",
    "key", "artifactDigest", "receiptId",
]
STATUSES = {"started", "succeeded", "retryable_failed", "terminal_failed"}
RECEIPT_NODES = {"register", "publish"}
SAFE_MAX = 9007199254740991


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest_array(values):
    raw = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def is_safe_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= SAFE_MAX


def nonempty_str(value):
    return isinstance(value, str) and len(value) > 0


def error(code):
    return JSONResponse(content={"error": code}, status_code=400)


def conflict(code):
    return JSONResponse(content={"error": code}, status_code=409)


def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS sessions(
                session TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                inputs TEXT NOT NULL,
                state TEXT NOT NULL
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS events(
                session TEXT NOT NULL,
                event_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                canonical TEXT NOT NULL,
                PRIMARY KEY(session, event_id)
            )
        """)
        # Cache is isolated by session.  Within a session it survives
        # revision changes and binds a successful content-addressed key to
        # its first artifact/event evidence permanently.
        c.execute("""
            CREATE TABLE IF NOT EXISTS cache(
                session TEXT NOT NULL,
                node TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                artifact_digest TEXT NOT NULL,
                event_id TEXT NOT NULL,
                PRIMARY KEY(session, node, cache_key)
            )
        """)
        c.commit()


init_db()


def load_session(c, session):
    row = c.execute(
        "SELECT revision, inputs, state FROM sessions WHERE session=?", (session,)
    ).fetchone()
    if row is None:
        return None
    return {
        "revision": row[0],
        "inputs": json.loads(row[1]),
        "state": json.loads(row[2]),
    }


def save_session(c, session, data):
    c.execute(
        "UPDATE sessions SET revision=?, inputs=?, state=? WHERE session=?",
        (
            data["revision"],
            compact(data["inputs"]),
            compact(data["state"]),
            session,
        ),
    )


def cache_get(c, session, node, key):
    if key is None:
        return None
    row = c.execute(
        "SELECT artifact_digest, event_id FROM cache "
        "WHERE session=? AND node=? AND cache_key=?",
        (session, node, key),
    ).fetchone()
    if row is None:
        return None
    return {"artifactDigest": row[0], "eventId": row[1]}


def cache_put(c, session, node, key, artifact, event_id):
    c.execute(
        "INSERT INTO cache(session,node,cache_key,artifact_digest,event_id) "
        "VALUES(?,?,?,?,?)",
        (session, node, key, artifact, event_id),
    )


def valid_inputs(inputs):
    return isinstance(inputs, dict) and all(
        nonempty_str(inputs.get(k)) for k in INPUT_KEYS
    )


def event_valid_shape(event):
    # Events with invalid shape/content are ignored, not request-level errors.
    if not isinstance(event, dict):
        return False
    if set(event.keys()) != set(EVENT_FIELDS):
        return False
    if not all(k in event for k in EVENT_FIELDS):
        return False
    if not nonempty_str(event["eventId"]):
        return False
    if not is_safe_int(event["revision"]):
        return False
    if event["node"] not in DAG:
        return False
    if not is_safe_int(event["attempt"]):
        return False
    if event["status"] not in STATUSES:
        return False
    if not nonempty_str(event["key"]):
        return False

    if event["status"] == "succeeded":
        if not nonempty_str(event["artifactDigest"]):
            return False
    elif event["artifactDigest"] is not None:
        return False

    if event["node"] in RECEIPT_NODES:
        if event["status"] == "succeeded":
            if event["receiptId"] != f"receipt:{event['node']}:{event['key']}":
                return False
        elif event["receiptId"] is not None:
            return False
    elif event["receiptId"] is not None:
        return False

    return True


def state_for_current_key(state, node, key):
    current = state.get(node)
    if current is None or current.get("key") != key:
        return {
            "status": None,
            "attempt": None,
            "eventId": None,
            "artifactDigest": None,
            "key": key,
        }
    return current


def dependency_values(node, inputs, reusable):
    # The dict insertion order follows NODE_DEPS exactly.
    if node == "verify_data":
        return {
            "generation": inputs["generation"],
            "checksum": inputs["checksum"],
        }
    if node == "prepare":
        return {
            "canonicalData": inputs["canonicalData"],
            "prepareCode": inputs["prepareCode"],
            "prepareConfig": inputs["prepareConfig"],
            "prepareArtifact": reusable["verify_data"]["artifactDigest"],
        }
    if node == "train":
        return {
            "prepareArtifact": reusable["prepare"]["artifactDigest"],
            "trainCode": inputs["trainCode"],
            "trainConfig": inputs["trainConfig"],
            "runtime": inputs["runtime"],
        }
    if node == "evaluate":
        return {
            "trainArtifact": reusable["train"]["artifactDigest"],
            "canonicalData": inputs["canonicalData"],
            "evaluateCode": inputs["evaluateCode"],
            "evaluateConfig": inputs["evaluateConfig"],
        }
    if node == "register":
        return {
            "evaluateArtifact": reusable["evaluate"]["artifactDigest"],
            "schemaDigest": inputs["schemaDigest"],
        }
    return {
        "registerArtifact": reusable["register"]["artifactDigest"],
        "publishConfig": inputs["publishConfig"],
    }


def compute_keys(c, session, inputs):
    """Walk the DAG and only create a downstream key after its parent is reusable."""
    keys = {}
    reusable = {}
    for node in DAG:
        parent = PARENT[node]
        if parent is not None and parent not in reusable:
            keys[node] = None
            continue

        vals = dependency_values(node, inputs, reusable)
        keys[node] = digest_array([vals[name] for name in NODE_DEPS[node]])
        hit = cache_get(c, session, node, keys[node])
        if hit is not None:
            reusable[node] = hit
    return keys, reusable


def transition(current, event):
    prev = current["status"]
    if prev is None:
        return "accept" if event["status"] == "started" and event["attempt"] == 1 else "ignore"

    if prev == "started":
        if event["attempt"] == current["attempt"] and event["status"] in {
            "succeeded", "retryable_failed", "terminal_failed"
        }:
            return "accept"
        return "conflict"

    if prev == "retryable_failed":
        if event["status"] == "started" and event["attempt"] == current["attempt"] + 1:
            return "accept"
        return "conflict"

    if prev == "terminal_failed":
        return "conflict"

    if prev == "succeeded":
        if event["status"] == "succeeded" and event["artifactDigest"] != current["artifactDigest"]:
            return "evidence"
        return "conflict"

    return "conflict"


def set_state(state, node, event):
    state[node] = {
        "status": event["status"],
        "attempt": event["attempt"],
        "eventId": event["eventId"],
        "artifactDigest": event["artifactDigest"],
        "key": event["key"],
    }


@app.post("/pipeline")
async def pipeline(request: Request):
    try:
        body = await request.json()
    except Exception:
        return error("INVALID_REQUEST")

    if not isinstance(body, dict) or set(body.keys()) != {"session", "revision", "inputs", "events"}:
        return error("INVALID_REQUEST")

    session = body["session"]
    revision = body["revision"]
    inputs = body["inputs"]
    events = body["events"]

    if not nonempty_str(session) or not is_safe_int(revision) or not valid_inputs(inputs) or not isinstance(events, list):
        return error("INVALID_REQUEST")

    with LOCK, sqlite3.connect(DB_PATH) as c:
        c.execute("BEGIN IMMEDIATE")
        s = load_session(c, session)

        if s is None:
            s = {"revision": revision, "inputs": deepcopy(inputs), "state": {}}
            c.execute(
                "INSERT INTO sessions(session,revision,inputs,state) VALUES(?,?,?,?)",
                (session, revision, compact(inputs), compact({})),
            )
        elif revision != s["revision"]:
            # Revision changes replace the active inputs and terminal/attempt
            # state. Successful content-addressed cache entries are retained.
            s = {"revision": revision, "inputs": deepcopy(inputs), "state": {}}
            save_session(c, session, s)
        elif compact(inputs) != compact(s["inputs"]):
            c.rollback()
            return conflict("REVISION_CONFLICT")

        working = deepcopy(s["state"])
        accepted = []
        ignored = []
        batch_ids = {}

        for event in events:
            eid = event.get("eventId") if isinstance(event, dict) else None

            if not event_valid_shape(event):
                if nonempty_str(eid):
                    ignored.append(eid)
                continue

            canonical = compact(event)

            # Duplicate IDs in this request are compared against their first
            # occurrence. A replay is ignored; a changed payload is atomic 409.
            if eid in batch_ids:
                if batch_ids[eid] != canonical:
                    c.rollback()
                    return conflict("EVENT_ID_CONFLICT")
                ignored.append(eid)
                continue
            batch_ids[eid] = canonical

            prior = c.execute(
                "SELECT canonical FROM events WHERE session=? AND event_id=?",
                (session, eid),
            ).fetchone()
            if prior is not None:
                if prior[0] == canonical:
                    ignored.append(eid)
                    continue
                c.rollback()
                return conflict("EVENT_ID_CONFLICT")

            # Wrong revision, wrong key, unavailable parent, etc. are ignored
            # and do not reserve the event ID.
            if event["revision"] != revision:
                ignored.append(eid)
                continue

            keys, reusable = compute_keys(c, session, inputs)
            expected_key = keys.get(event["node"])
            if expected_key is None or event["key"] != expected_key:
                ignored.append(eid)
                continue

            # A successful cache entry is immutable evidence, even if it came
            # from an earlier revision in this same session.
            cached = cache_get(c, session, event["node"], expected_key)
            if cached is not None:
                if event["status"] == "succeeded" and event["artifactDigest"] != cached["artifactDigest"]:
                    c.rollback()
                    return conflict("EVIDENCE_CONFLICT")
                c.rollback() if False else None
                # Any new event against an already-successful current cache is
                # a status conflict, except the differing-artifact evidence
                # case handled above.
                c.rollback()
                return conflict("STATUS_CONFLICT")

            parent = PARENT[event["node"]]
            if parent is not None and parent not in reusable:
                ignored.append(eid)
                continue

            current = state_for_current_key(working, event["node"], expected_key)
            tr = transition(current, event)
            if tr == "ignore":
                ignored.append(eid)
                continue
            if tr == "evidence":
                c.rollback()
                return conflict("EVIDENCE_CONFLICT")
            if tr == "conflict":
                c.rollback()
                return conflict("STATUS_CONFLICT")

            set_state(working, event["node"], event)
            c.execute(
                "INSERT INTO events(session,event_id,revision,canonical) VALUES(?,?,?,?)",
                (session, eid, revision, canonical),
            )

            if event["status"] == "succeeded":
                cache_put(c, session, event["node"], expected_key, event["artifactDigest"], eid)

            accepted.append(eid)

        s["state"] = working
        save_session(c, session, s)
        c.commit()

        # Read the committed state back and construct deterministic response.
        with sqlite3.connect(DB_PATH) as rc:
            keys, reusable = compute_keys(rc, session, inputs)
            nodes = []

            for node in DAG:
                key = keys[node]
                deps = {}
                if key is not None:
                    vals = dependency_values(node, inputs, reusable)
                    for dep_name in NODE_DEPS[node]:
                        deps[dep_name] = vals[dep_name]
                    deps["cacheKey"] = key

                cur = state_for_current_key(working, node, key)
                hit = cache_get(rc, session, node, key)

                if hit is not None:
                    action = "reuse"
                    reason = "CACHE_HIT"
                    triggers = [hit["eventId"]]
                elif cur["status"] == "terminal_failed":
                    action = "block"
                    reason = "TERMINAL_FAILURE"
                    triggers = [cur["eventId"]]
                elif cur["status"] == "started":
                    action = "block"
                    reason = "RUNNING"
                    triggers = [cur["eventId"]]
                elif cur["status"] == "retryable_failed":
                    action = "rerun"
                    reason = "RETRYABLE_FAILURE"
                    triggers = [cur["eventId"]]
                elif key is None:
                    parent = PARENT[node]
                    if parent is not None and state_for_current_key(working, parent, keys.get(parent)).get("status") == "terminal_failed":
                        action = "block"
                        reason = "UPSTREAM_TERMINAL"
                    else:
                        action = "block"
                        reason = "UPSTREAM_PENDING"
                    triggers = []
                else:
                    action = "rerun"
                    reason = "CACHE_MISS"
                    triggers = []

                nodes.append({
                    "node": node,
                    "action": action,
                    "reasonCodes": [reason],
                    "dependencyDigests": deps,
                    "triggeringEventIds": triggers,
                })

        return {
            "revision": revision,
            "acceptedEventIds": accepted,
            "ignoredEventIds": ignored,
            "nodes": nodes,
        }


@app.get("/health")
def health():
    return {"status": "ok"}
