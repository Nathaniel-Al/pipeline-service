from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any, Optional

DB_LOCK = threading.RLock()

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

# Inputs required for each node's cache key
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


@dataclass
class SessionState:
    """Complete state for a session"""
    session: str
    revision: int
    inputs: dict[str, str]
    nodes: dict[str, NodeState]
    event_ids_seen: set[str]
    event_canonical: dict[str, str]


def get_db() -> sqlite3.Connection:
    """Get database connection"""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        "session TEXT PRIMARY KEY, "
        "revision INTEGER NOT NULL, "
        "inputs TEXT NOT NULL, "
        "node_states TEXT NOT NULL, "
        "event_ids_seen TEXT NOT NULL, "
        "event_canonical TEXT NOT NULL)"
    )
    conn.commit()
    return conn


def sha256_compact(data: list[str]) -> str:
    """Compute SHA-256 of compact JSON array"""
    compact = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(compact.encode("utf-8", "strict")).hexdigest()


def compute_cache_key(node: str, inputs: dict[str, str], artifact_map: dict[str, Optional[str]]) -> Optional[str]:
    """
    Compute cache key for a node.
    Returns None if parent artifacts are not available.
    artifact_map is current state: node -> artifact_digest or None
    """
    required = NODE_INPUTS[node]
    values = []
    
    for key in required:
        # If key is a direct input, use it
        if key in inputs:
            values.append(inputs[key])
        # If key is an artifact reference (ends with "Artifact"), resolve it
        elif key.endswith("Artifact"):
            # Extract parent node name (e.g., "prepareArtifact" -> "prepare")
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
    """Check if x is a safe non-negative integer"""
    return isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 9007199254740991


def is_nonempty_string(x: Any) -> bool:
    """Check if x is a non-empty string"""
    return isinstance(x, str) and len(x) > 0


def validate_event(event: Any) -> tuple[bool, Optional[str]]:
    """
    Validate event structure.
    Returns (is_valid, error_code)
    """
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
    
    # Ignore if node doesn't exist (shouldn't happen if validated)
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
    node_state = session_state.nodes.get(node)
    if node_state is None:
        return False, None
    
    # Check parent availability (only for transitions that need it)
    if node not in ["verify_data"]:
        parent_keys = DAG_PARENTS[node]
        for parent in parent_keys:
            parent_state = session_state.nodes[parent]
            if parent_state.status != "succeeded":
                # Parent not available/successful
                return False, None
    
    # State transition logic
    current_status = node_state.status
    current_attempt = node_state.attempt
    
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
        elif attempt < current_attempt and current_status not in ["none", None]:
            # Lower attempt: ignore
            return False, None
        else:
            # STATUS_CONFLICT
            return False, "STATUS_CONFLICT"
    
    elif status in ["succeeded", "retryable_failed", "terminal_failed"]:
        if current_status == "started" and attempt == current_attempt:
            # Accept: completion of current attempt
            if status == "succeeded":
                # Check for evidence conflict: different artifact for same key
                if node_state.artifact_digest is not None and node_state.artifact_digest != artifact_digest:
                    return False, "EVIDENCE_CONFLICT"
                node_state.artifact_digest = artifact_digest
                node_state.receipt_id = event.get("receiptId")
            
            node_state.status = status
            node_state.event_id = event_id
            session_state.event_ids_seen.add(event_id)
            session_state.event_canonical[event_id] = event_canonical
            return True, None
        elif current_status == "succeeded" and status == "succeeded":
            # Same success, different artifact
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
    """Load session state from database"""
    row = db.execute(
        "SELECT revision, inputs, node_states, event_ids_seen, event_canonical FROM sessions WHERE session = ?",
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
    """Save session state to database"""
    node_states_json = json.dumps({
        n: {
            "status": state.nodes[n].status,
            "attempt": state.nodes[n].attempt,
            "artifact_digest": state.nodes[n].artifact_digest,
            "event_id": state.nodes[n].event_id,
            "receipt_id": state.nodes[n].receipt_id,
        }
        for n in DAG_NODES
    }, separators=(",", ":"))
    
    event_ids_json = json.dumps(sorted(state.event_ids_seen), separators=(",", ":"))
    event_canonical_json = json.dumps(state.event_canonical, separators=(",", ":"))
    
    db.execute(
        "INSERT OR REPLACE INTO sessions (session, revision, inputs, node_states, event_ids_seen, event_canonical) VALUES (?, ?, ?, ?, ?, ?)",
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
    """Process pipeline request"""
    db = get_db()
    
    # Validate request
    is_valid, error_code = validate_request(body)
    if not is_valid:
        return {"error": error_code}
    
    session_str = body["session"]
    new_revision = body["revision"]
    new_inputs = body["inputs"]
    events = body["events"]
    
    with DB_LOCK:
        # Load existing session
        session_state = load_session(db, session_str)
        
        # Check for revision conflict
        if session_state.revision != 0:  # Session exists
            if new_revision < session_state.revision:
                # Cannot go backwards
                return {"error": "REVISION_CONFLICT"}
            
            if new_revision == session_state.revision:
                # Same revision: check inputs match exactly
                if new_inputs != session_state.inputs:
                    return {"error": "REVISION_CONFLICT"}
                # Inputs match: reuse this revision's data (no state changes)
            elif new_revision == session_state.revision + 1:
                # New revision: replace inputs, clear event tracking, keep successful artifacts
                session_state.revision = new_revision
                session_state.inputs = new_inputs
                # Clear event IDs for new revision
                session_state.event_ids_seen = set()
                session_state.event_canonical = {}
                # Reset all nodes but keep successful artifacts
                for n in DAG_NODES:
                    old_state = session_state.nodes[n]
                    if old_state.status != "succeeded":
                        session_state.nodes[n] = NodeState()
                    else:
                        # Keep artifact but clear transient state
                        session_state.nodes[n] = NodeState(
                            status="succeeded",
                            artifact_digest=old_state.artifact_digest,
                            event_id=old_state.event_id,
                            receipt_id=old_state.receipt_id,
                        )
            else:
                # Gap in revisions
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
    for node in DAG_NODES:
        node_state = session_state.nodes[node]
        cache_key = compute_cache_key(node, session_state.inputs, artifact_map)
        
        # Determine action and reason
        action = "block"
        reason_codes = []
        triggering_event_ids = []
        
        if node_state.status == "succeeded":
            action = "reuse"
            reason_codes = ["CACHE_HIT"]
            if node_state.event_id:
                triggering_event_ids = [node_state.event_id]
        elif node_state.status == "terminal_failed":
            action = "block"
            reason_codes = ["TERMINAL_FAILURE"]
            if node_state.event_id:
                triggering_event_ids = [node_state.event_id]
        elif node_state.status == "started":
            action = "block"
            reason_codes = ["RUNNING"]
            if node_state.event_id:
                triggering_event_ids = [node_state.event_id]
        elif cache_key is None:
            # Parent not ready
            action = "block"
            reason_codes = ["UPSTREAM_PENDING"]
        else:
            # Ready to run
            action = "rerun"
            if node_state.status is None:
                reason_codes = ["CACHE_MISS"]
            else:
                reason_codes = ["RETRYABLE_FAILURE"]
        
        # Build dependency digests - include ALL keys from NODE_INPUTS
        dependency_digests = {}
        required = NODE_INPUTS[node]
        for key in required:
            if key in session_state.inputs:
                dependency_digests[key] = session_state.inputs[key]
            elif key.endswith("Artifact"):
                parent_node = key.replace("Artifact", "")
                artifact = artifact_map.get(parent_node)
                if artifact is not None:
                    dependency_digests[key] = artifact
        
        if cache_key is not None:
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
