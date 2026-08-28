from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Optional

# Global database connection and lock
DB_LOCK = threading.RLock()
DB_PATH = os.environ.get("PIPELINE_DB", "./data/pipeline_state.sqlite3")
_db_instance = None

# DAG structure: fixed ordering with dependencies
DAG_NODES = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
DAG_PARENTS = {
    "verify_data": [],
    "prepare": ["verify_data"],
    "train": ["prepare"],
    "evaluate": ["train"],
    "register": ["evaluate"],
    "publish": ["register"],
}

# Inputs required for each node's cache key (ORDER MATTERS)
NODE_INPUTS = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig"],
    "train": ["prepareArtifact", "trainCode", "trainConfig", "runtime"],
    "evaluate": ["trainArtifact", "canonicalData", "evaluateCode", "evaluateConfig"],
    "register": ["evaluateArtifact", "schemaDigest"],
    "publish": ["registerArtifact", "publishConfig"],
}


@dataclass
class NodeState:
    """State of a single node within a session"""
    status: Optional[str] = None  # None, "started", "succeeded", "retryable_failed", "terminal_failed"
    attempt: int = 0
    artifact_digest: Optional[str] = None
    event_id: Optional[str] = None
    receipt_id: Optional[str] = None
    cache_key: Optional[str] = None  # Immutable cache key once succeeded


@dataclass
class SessionState:
    """Complete state for a session"""
    session: str
    revision: int
    inputs: dict[str, str]
    nodes: dict[str, NodeState]
    event_ids_seen: set[str]
    event_canonical: dict[str, str]


def init_db() -> None:
    """Initialize database with persistent storage"""
    global _db_instance
    db_dir = os.path.dirname(DB_PATH)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    _db_instance = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    _db_instance.execute(
        "CREATE TABLE IF NOT EXISTS pipeline_sessions ("
        "session TEXT PRIMARY KEY, "
        "revision INTEGER NOT NULL, "
        "inputs TEXT NOT NULL, "
        "node_states TEXT NOT NULL, "
        "event_ids_seen TEXT NOT NULL, "
        "event_canonical TEXT NOT NULL)"
    )
    _db_instance.commit()


def get_db() -> sqlite3.Connection:
    """Get global database connection"""
    if _db_instance is None:
        init_db()
    return _db_instance


def sha256_compact(data: list[str]) -> str:
    """Compute SHA-256 of compact JSON array (preserves order)"""
    compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(compact.encode("utf-8", "strict")).hexdigest()


def compute_cache_key(node: str, inputs: dict[str, str], artifact_map: dict[str, Optional[str]]) -> Optional[str]:
    """
    Compute cache key for a node.
    Returns None if parent artifacts are not available.
    ORDER MATTERS: must match NODE_INPUTS order exactly.
    """
    required = NODE_INPUTS[node]
    values = []
    
    for key in required:
        if key in inputs:
            # Direct input
            values.append(inputs[key])
        elif key.endswith("Artifact"):
            # Artifact reference: must be from a succeeded parent
            parent_node = key.replace("Artifact", "")
            artifact = artifact_map.get(parent_node)
            if artifact is None:
                # Parent not available
                return None
            values.append(artifact)
        else:
            # Missing required input
            return None
    
    return sha256_compact(values)


def compact_json(obj: Any) -> str:
    """Compact JSON representation"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def is_safe_nonnegative_integer(x: Any) -> bool:
    """Check if x is a safe non-negative integer (0 to 2^53-1)"""
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 9007199254740991


def is_nonempty_string(x: Any) -> bool:
    """Check if x is a non-empty string"""
    return isinstance(x, str) and len(x) > 0


def validate_event(event: Any) -> tuple[bool, Optional[str]]:
    """Validate event structure"""
    if not isinstance(event, dict):
        return False, "INVALID_EVENT"
    
    required = {"eventId", "revision", "node", "attempt", "status", "key", "artifactDigest", "receiptId"}
    if set(event.keys()) != required:
        return False, "INVALID_EVENT"
    
    if not is_nonempty_string(event.get("eventId")):
        return False, "INVALID_EVENT"
    if not is_safe_nonnegative_integer(event.get("revision")) or event["revision"] < 1:
        return False, "INVALID_EVENT"
    if event.get("node") not in DAG_NODES:
        return False, "INVALID_EVENT"
    if not is_safe_nonnegative_integer(event.get("attempt")) or event["attempt"] < 1:
        return False, "INVALID_EVENT"
    if event.get("status") not in ["started", "succeeded", "retryable_failed", "terminal_failed"]:
        return False, "INVALID_EVENT"
    if not is_nonempty_string(event.get("key")):
        return False, "INVALID_EVENT"
    
    # Artifact digest: non-empty string for success, null otherwise
    status = event["status"]
    artifact = event.get("artifactDigest")
    if status == "succeeded":
        if not is_nonempty_string(artifact):
            return False, "INVALID_EVENT"
    else:
        if artifact is not None:
            return False, "INVALID_EVENT"
    
    # Receipt: receipt:<node>:<key> for register/publish success, null otherwise
    receipt = event.get("receiptId")
    node = event["node"]
    if node in ["register", "publish"] and status == "succeeded":
        expected_receipt = f"receipt:{node}:{event['key']}"
        if receipt != expected_receipt:
            return False, "INVALID_EVENT"
    else:
        if receipt is not None:
            return False, "INVALID_EVENT"
    
    return True, None


def is_parent_ready(node: str, session_state: SessionState) -> bool:
    """Check if all parents of a node are succeeded"""
    if node not in DAG_PARENTS:
        return True
    for parent in DAG_PARENTS[node]:
        if session_state.nodes[parent].status != "succeeded":
            return False
    return True


def process_event(
    session_state: SessionState,
    event: dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """
    Process a single event for the session.
    Returns (should_accept, error_code)
    """
    event_id = event["eventId"]
    event_revision = event["revision"]
    node = event["node"]
    attempt = event["attempt"]
    status = event["status"]
    key = event["key"]
    artifact_digest = event.get("artifactDigest")
    
    # Ignore if wrong revision
    if event_revision != session_state.revision:
        return False, None
    
    # Ignore if node doesn't exist
    if node not in DAG_NODES:
        return False, None
    
    # Event ID conflict: same ID with different canonical form
    event_canonical = compact_json(event)
    if event_id in session_state.event_ids_seen:
        if session_state.event_canonical.get(event_id) != event_canonical:
            return False, "EVENT_ID_CONFLICT"
        # Exact replay: ignore
        return False, None
    
    # Get current node state
    node_state = session_state.nodes[node]
    current_status = node_state.status
    current_attempt = node_state.attempt
    
    # Parent availability check (critical for all transitions)
    if not is_parent_ready(node, session_state):
        # Ignore: parent not available
        return False, None
    
    # Validate key matches expected cache key (only if we can compute it)
    artifact_map = {n: session_state.nodes[n].artifact_digest for n in DAG_NODES}
    expected_key = compute_cache_key(node, session_state.inputs, artifact_map)
    if expected_key is not None and key != expected_key:
        # Key mismatch: ignore
        return False, None
    
    # State transition logic
    if status == "started":
        if attempt == 1 and current_status is None:
            # Accept: first start
            node_state.status = "started"
            node_state.attempt = 1
            node_state.event_id = event_id
            session_state.event_ids_seen.add(event_id)
            session_state.event_canonical[event_id] = event_canonical
            return True, None
        elif attempt == current_attempt + 1 and current_status == "retryable_failed":
            # Accept: retry after retryable failure
            node_state.status = "started"
            node_state.attempt = attempt
            node_state.event_id = event_id
            node_state.artifact_digest = None
            node_state.receipt_id = None
            session_state.event_ids_seen.add(event_id)
            session_state.event_canonical[event_id] = event_canonical
            return True, None
        elif attempt < current_attempt:
            # Lower attempt: ignore
            return False, None
        else:
            # STATUS_CONFLICT
            return False, "STATUS_CONFLICT"
    
    elif status in ["succeeded", "retryable_failed", "terminal_failed"]:
        if current_status == "started" and attempt == current_attempt:
            # Accept: completion of current attempt
            if status == "succeeded":
                # Immutable evidence: bind artifact to key permanently
                if node_state.artifact_digest is not None and node_state.artifact_digest != artifact_digest:
                    return False, "EVIDENCE_CONFLICT"
                node_state.artifact_digest = artifact_digest
                node_state.cache_key = key  # Remember the cache key
                node_state.receipt_id = event.get("receiptId")
            
            node_state.status = status
            node_state.event_id = event_id
            session_state.event_ids_seen.add(event_id)
            session_state.event_canonical[event_id] = event_canonical
            return True, None
        elif current_status == "succeeded" and status == "succeeded":
            # Already succeeded: check for evidence conflict
            if node_state.artifact_digest != artifact_digest:
                return False, "EVIDENCE_CONFLICT"
            # Same artifact, same status: ignore (replay)
            return False, None
        elif current_status == "succeeded":
            # Any other new event after success
            return False, "STATUS_CONFLICT"
        elif current_status == "terminal_failed":
            # Any new event after terminal failure
            return False, "STATUS_CONFLICT"
        elif current_status is None:
            # Completion without prior start
            return False, None
        else:
            # Wrong attempt
            return False, None
    
    return False, None


def load_session(db: sqlite3.Connection, session: str) -> SessionState:
    """Load session state from persistent database"""
    row = db.execute(
        "SELECT revision, inputs, node_states, event_ids_seen, event_canonical FROM pipeline_sessions WHERE session = ?",
        (session,),
    ).fetchone()
    
    if row is None:
        # New session
        state = SessionState(
            session=session,
            revision=0,
            inputs={},
            nodes={n: NodeState() for n in DAG_NODES},
            event_ids_seen=set(),
            event_canonical={},
        )
    else:
        revision, inputs_json, node_states_json, event_ids_json, event_canonical_json = row
        inputs = json.loads(inputs_json)
        node_states_raw = json.loads(node_states_json)
        event_ids_seen = set(json.loads(event_ids_json))
        event_canonical = json.loads(event_canonical_json)
        
        nodes = {}
        for n in DAG_NODES:
            ns_raw = node_states_raw.get(n, {})
            nodes[n] = NodeState(
                status=ns_raw.get("status"),
                attempt=ns_raw.get("attempt", 0),
                artifact_digest=ns_raw.get("artifact_digest"),
                event_id=ns_raw.get("event_id"),
                receipt_id=ns_raw.get("receipt_id"),
                cache_key=ns_raw.get("cache_key"),
            )
        
        state = SessionState(
            session=session,
            revision=revision,
            inputs=inputs,
            nodes=nodes,
            event_ids_seen=event_ids_seen,
            event_canonical=event_canonical,
        )
    
    return state


def save_session(db: sqlite3.Connection, state: SessionState) -> None:
    """Save session state to persistent database"""
    node_states_json = json.dumps({
        n: {
            "status": state.nodes[n].status,
            "attempt": state.nodes[n].attempt,
            "artifact_digest": state.nodes[n].artifact_digest,
            "event_id": state.nodes[n].event_id,
            "receipt_id": state.nodes[n].receipt_id,
            "cache_key": state.nodes[n].cache_key,
        }
        for n in DAG_NODES
    }, separators=(",", ":"))
    
    event_ids_json = json.dumps(sorted(state.event_ids_seen), separators=(",", ":"))
    event_canonical_json = json.dumps(state.event_canonical, separators=(",", ":"))
    
    db.execute(
        "INSERT OR REPLACE INTO pipeline_sessions (session, revision, inputs, node_states, event_ids_seen, event_canonical) VALUES (?, ?, ?, ?, ?, ?)",
        (
            state.session,
            state.revision,
            json.dumps(state.inputs, separators=(",", ":")),
            node_states_json,
            event_ids_json,
            event_canonical_json,
        ),
    )
    db.commit()


def validate_request(body: Any) -> tuple[bool, Optional[str]]:
    """Validate request structure"""
    if not isinstance(body, dict):
        return False, "INVALID_REQUEST"
    
    if not is_nonempty_string(body.get("session")):
        return False, "INVALID_REQUEST"
    if not is_safe_nonnegative_integer(body.get("revision")) or body.get("revision") < 1:
        return False, "INVALID_REQUEST"
    
    inputs = body.get("inputs")
    if not isinstance(inputs, dict):
        return False, "INVALID_REQUEST"
    
    required_inputs = {
        "generation", "checksum", "canonicalData", "prepareCode", "prepareConfig",
        "trainCode", "trainConfig", "runtime", "evaluateCode", "evaluateConfig",
        "schemaDigest", "publishConfig"
    }
    for key in required_inputs:
        if not is_nonempty_string(inputs.get(key)):
            return False, "INVALID_REQUEST"
    
    events = body.get("events")
    if not isinstance(events, list):
        return False, "INVALID_REQUEST"
    
    return True, None


def process_pipeline(body: dict[str, Any]) -> dict[str, Any]:
    """Process pipeline request with proper state management"""
    # Validate request
    is_valid, error_code = validate_request(body)
    if not is_valid:
        return {"error": error_code}
    
    session_str = body["session"]
    new_revision = body["revision"]
    new_inputs = body["inputs"]
    events = body["events"]
    
    db = get_db()
    
    with DB_LOCK:
        # Load existing session
        session_state = load_session(db, session_str)
        
        # Check for revision conflict
        if session_state.revision != 0:  # Session exists
            if new_revision <= session_state.revision:
                # Invalid revision number: must be strictly increasing or same
                return {"error": "REVISION_CONFLICT"}
            
            if new_revision == session_state.revision + 1:
                # New revision: validate inputs are different (if different revision expected)
                # Replace inputs and clear attempt/terminal state, but KEEP succeeded artifacts
                session_state.revision = new_revision
                session_state.inputs = new_inputs
                # Clear non-terminal state
                for n in DAG_NODES:
                    old_state = session_state.nodes[n]
                    if old_state.status == "succeeded":
                        # Keep successful artifacts: they're immutable evidence
                        pass
                    else:
                        # Clear all transient state
                        session_state.nodes[n] = NodeState()
            else:
                # Revision jump > 1 is invalid
                return {"error": "REVISION_CONFLICT"}
        else:
            # New session
            session_state.revision = new_revision
            session_state.inputs = new_inputs
        
        # Process events in order
        accepted_event_ids = []
        ignored_event_ids = []
        
        for event in events:
            # Validate event
            is_valid, error_code = validate_event(event)
            if not is_valid:
                return {"error": error_code}
            
            # Process event
            should_accept, error_code = process_event(session_state, event)
            if error_code is not None:
                # Conflict detected: rollback entire batch
                return {"error": error_code}
            
            if should_accept:
                accepted_event_ids.append(event["eventId"])
            else:
                ignored_event_ids.append(event["eventId"])
        
        # Save session state
        save_session(db, session_state)
    
    # Build response
    artifact_map = {n: session_state.nodes[n].artifact_digest for n in DAG_NODES}
    
    response_nodes = []
    has_terminal_failure = False
    
    for node in DAG_NODES:
        node_state = session_state.nodes[node]
        cache_key = compute_cache_key(node, session_state.inputs, artifact_map)
        
        # Determine action and reason
        action = "block"
        reason_codes = []
        triggering_event_ids = []
        
        # Check if ancestor had terminal failure
        if has_terminal_failure:
            action = "block"
            reason_codes = ["UPSTREAM_TERMINAL"]
        elif node_state.status == "succeeded":
            action = "reuse"
            reason_codes = ["CACHE_HIT"]
            if node_state.event_id:
                triggering_event_ids = [node_state.event_id]
        elif node_state.status == "terminal_failed":
            action = "block"
            reason_codes = ["TERMINAL_FAILURE"]
            if node_state.event_id:
                triggering_event_ids = [node_state.event_id]
            has_terminal_failure = True
        elif node_state.status == "started":
            action = "block"
            reason_codes = ["RUNNING"]
            if node_state.event_id:
                triggering_event_ids = [node_state.event_id]
        elif not is_parent_ready(node, session_state):
            # Parent not ready
            action = "block"
            reason_codes = ["UPSTREAM_PENDING"]
        elif cache_key is None:
            # Can't compute cache key (shouldn't happen if parents ready)
            action = "block"
            reason_codes = ["UPSTREAM_PENDING"]
        else:
            # Ready to run
            action = "rerun"
            if node_state.status is None:
                reason_codes = ["CACHE_MISS"]
            else:
                reason_codes = ["RETRYABLE_FAILURE"]
        
        # Build dependency digests (in NODE_INPUTS order)
        dependency_digests = {}
        required = NODE_INPUTS[node]
        for key in required:
            if key in session_state.inputs:
                dependency_digests[key] = session_state.inputs[key]
            elif key.endswith("Artifact"):
                parent_node = key.replace("Artifact", "")
                artifact = artifact_map.get(parent_node)
                if artifact:
                    dependency_digests[key] = artifact
        
        if cache_key:
            dependency_digests["cacheKey"] = cache_key
        
        response_nodes.append({
            "node": node,
            "action": action,
            "reasonCodes": reason_codes,
            "dependencyDigests": dependency_digests,
            "triggeringEventIds": triggering_event_ids,
        })
    
    return {
        "revision": session_state.revision,
        "acceptedEventIds": accepted_event_ids,
        "ignoredEventIds": ignored_event_ids,
        "nodes": response_nodes,
    }


# Initialize database on module load
init_db()
