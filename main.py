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
if not os.path.isabs(DB_PATH):
    DB_PATH = os.path.abspath(DB_PATH)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
LOCK = threading.RLock()

DAG = ["verify_data", "prepare", "train", "evaluate", "register", "publish"]
PARENT = {
    "verify_data": None, "prepare": "verify_data", "train": "prepare",
    "evaluate": "train", "register": "evaluate", "publish": "register"
}
NODE_DEPS = {
    "verify_data": ["generation", "checksum"],
    "prepare": ["canonicalData", "prepareCode", "prepareConfig", "prepareArtifact"],
    "train": ["prepareArtifact", "trainCode", "trainConfig", "runtime"],
    "evaluate": ["trainArtifact", "canonicalData", "evaluateCode", "evaluateConfig"],
    "register": ["evaluateArtifact", "schemaDigest"],
    "publish": ["registerArtifact", "publishConfig"],
}
INPUT_KEYS = [
    "generation","checksum","canonicalData","prepareCode","prepareConfig",
    "trainCode","trainConfig","runtime","evaluateCode","evaluateConfig",
    "schemaDigest","publishConfig"
]
EVENT_FIELDS = ["eventId","revision","node","attempt","status","key","artifactDigest","receipt"]
STATUSES = {"started","succeeded","retryable_failed","terminal_failed"}
RECEIPT_NODES = {"register","publish"}

def compact(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def digest_array(values):
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

def is_safe_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and 0 < x <= 9007199254740991

def nonempty_str(x):
    return isinstance(x, str) and len(x) > 0

def canonical_json(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

def init_db():
    with sqlite3.connect(DB_PATH) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS sessions(
            session TEXT PRIMARY KEY, revision INTEGER, inputs TEXT NOT NULL,
            state TEXT NOT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS events(
            session TEXT NOT NULL, event_id TEXT NOT NULL,
            revision INTEGER NOT NULL, canonical TEXT NOT NULL,
            PRIMARY KEY(session,event_id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
            session TEXT NOT NULL, node TEXT NOT NULL, cache_key TEXT NOT NULL,
            artifact_digest TEXT NOT NULL, event_id TEXT NOT NULL,
            PRIMARY KEY(session,node,cache_key)
        )""")
        c.commit()

init_db()

def load_session(c, session):
    row=c.execute("SELECT revision,inputs,state FROM sessions WHERE session=?", (session,)).fetchone()
    if not row: return None
    return {"revision":row[0], "inputs":json.loads(row[1]), "state":json.loads(row[2])}

def save_session(c, session, data):
    c.execute("UPDATE sessions SET revision=?,inputs=?,state=? WHERE session=?",
              (data["revision"], canonical_json(data["inputs"]), canonical_json(data["state"]), session))

def cache_get(c, session, node, key):
    if key is None: return None
    row=c.execute("SELECT artifact_digest,event_id FROM cache WHERE session=? AND node=? AND cache_key=?",
                  (session,node,key)).fetchone()
    return None if not row else {"artifactDigest":row[0],"eventId":row[1]}

def cache_put(c, session, node, key, artifact, event_id):
    c.execute("INSERT OR IGNORE INTO cache(session,node,cache_key,artifact_digest,event_id) VALUES(?,?,?,?,?)",
              (session,node,key,artifact,event_id))

def valid_inputs(inputs):
    if not isinstance(inputs, dict): return False
    return all(nonempty_str(inputs.get(k)) for k in INPUT_KEYS)

def request_error(code):
    return JSONResponse({"error": code}, status_code=400)

def conflict(code):
    return JSONResponse({"error": code}, status_code=409)

def event_valid_shape(e):
    if not isinstance(e, dict) or list(e.keys()) != EVENT_FIELDS:
        return False
    if not nonempty_str(e["eventId"]): return False
    if not is_safe_int(e["revision"]): return False
    if e["node"] not in DAG: return False
    if not is_safe_int(e["attempt"]): return False
    if e["status"] not in STATUSES: return False
    if not nonempty_str(e["key"]): return False
    if e["status"] == "succeeded":
        if not nonempty_str(e["artifactDigest"]): return False
    elif e["artifactDigest"] is not None:
        return False
    if e["node"] in RECEIPT_NODES:
        if e["status"] == "succeeded":
            if e["receipt"] != f"receipt:{e['node']}:{e['key']}": return False
        elif e["receipt"] is not None: return False
    elif e["receipt"] is not None:
        return False
    return True

def dependency_values(node, inputs, reusable):
    vals={}
    if node=="verify_data":
        vals={"generation":inputs["generation"],"checksum":inputs["checksum"]}
    elif node=="prepare":
        vals={"canonicalData":inputs["canonicalData"],"prepareCode":inputs["prepareCode"],
              "prepareConfig":inputs["prepareConfig"],"prepareArtifact":reusable["verify_data"]["artifactDigest"]}
    elif node=="train":
        vals={"prepareArtifact":reusable["prepare"]["artifactDigest"],"trainCode":inputs["trainCode"],
              "trainConfig":inputs["trainConfig"],"runtime":inputs["runtime"]}
    elif node=="evaluate":
        vals={"trainArtifact":reusable["train"]["artifactDigest"],"canonicalData":inputs["canonicalData"],
              "evaluateCode":inputs["evaluateCode"],"evaluateConfig":inputs["evaluateConfig"]}
    elif node=="register":
        vals={"evaluateArtifact":reusable["evaluate"]["artifactDigest"],"schemaDigest":inputs["schemaDigest"]}
    elif node=="publish":
        vals={"registerArtifact":reusable["register"]["artifactDigest"],"publishConfig":inputs["publishConfig"]}
    return vals

def compute_keys(c, session, inputs, state):
    """Returns per-node key and the immutable reusable evidence, walking the DAG."""
    keys={}
    reusable={}
    for node in DAG:
        parent=PARENT[node]
        if parent is not None and parent not in reusable:
            keys[node]=None
            continue
        vals=dependency_values(node,inputs,reusable)
        key=digest_array([vals[k] for k in NODE_DEPS[node]])
        keys[node]=key
        hit=cache_get(c,session,node,key)
        # A current success is also reusable, and is represented in cache.
        if hit:
            reusable[node]=hit
    return keys,reusable

def node_state(state,node):
    return state.get(node, {"status":None,"attempt":None,"eventId":None,"artifactDigest":None,"key":None})

def transition(state,node,e):
    cur=node_state(state,node)
    prev=cur["status"]
    if prev is None:
        if e["status"]=="started" and e["attempt"]==1:
            return "accept"
        return "ignore"
    if prev=="started":
        if e["attempt"]==cur["attempt"] and e["status"] in {"succeeded","retryable_failed","terminal_failed"}:
            return "accept"
        return "conflict"
    if prev=="retryable_failed":
        if e["status"]=="started" and e["attempt"]==cur["attempt"]+1:
            return "accept"
        return "conflict"
    if prev=="terminal_failed":
        return "conflict"
    if prev=="succeeded":
        if e["status"]=="succeeded" and e["artifactDigest"]!=cur["artifactDigest"]:
            return "evidence"
        return "conflict"
    return "conflict"

def set_state(state,node,e):
    state[node]={
        "status":e["status"], "attempt":e["attempt"], "eventId":e["eventId"],
        "artifactDigest":e["artifactDigest"], "key":e["key"]
    }

@app.post("/pipeline")
async def pipeline(request: Request):
    try:
        body=await request.json()
    except Exception:
        return request_error("INVALID_REQUEST")
    if not isinstance(body,dict) or set(body.keys()) != {"session","revision","inputs","events"}:
        return request_error("INVALID_REQUEST")
    session=body["session"]
    revision=body["revision"]
    inputs=body["inputs"]
    events=body["events"]
    if not nonempty_str(session) or not is_safe_int(revision) or not valid_inputs(inputs) or not isinstance(events,list):
        return request_error("INVALID_REQUEST")

    with LOCK, sqlite3.connect(DB_PATH) as c:
        c.execute("BEGIN IMMEDIATE")
        s=load_session(c,session)
        if s is None:
            s={"revision":revision,"inputs":deepcopy(inputs),
               "state":{}}
            c.execute("INSERT INTO sessions(session,revision,inputs,state) VALUES(?,?,?,?)",
                      (session,revision,canonical_json(inputs),canonical_json({})))
        elif revision != s["revision"]:
            # A different revision replaces state and inputs, but only after
            # confirming the request is internally well formed.
            if revision < s["revision"]:
                c.rollback()
                return conflict("REVISION_CONFLICT")
            s={"revision":revision,"inputs":deepcopy(inputs),"state":{}}
            save_session(c,session,s)

        # Same revision must have byte-for-byte canonical input equality.
        else:
            if canonical_json(inputs) != canonical_json(s["inputs"]):
                c.rollback()
                return conflict("REVISION_CONFLICT")

        # For a new revision, state has been reset; cache remains.
        accepted=[]; ignored=[]
        working=deepcopy(s["state"])
        seen_in_batch=set()

        # Recompute expected keys against current working state after each event.
        for e in events:
            if not event_valid_shape(e):
                ignored.append(e.get("eventId") if isinstance(e,dict) and nonempty_str(e.get("eventId")) else None)
                continue
            eid=e["eventId"]
            if eid in seen_in_batch:
                # compare with first canonical representation
                prior=next(x for x in events[:events.index(e)] if isinstance(x,dict) and x.get("eventId")==eid)
                if canonical_json(prior)!=canonical_json(e):
                    c.rollback(); return conflict("EVENT_ID_CONFLICT")
                ignored.append(eid); continue
            seen_in_batch.add(eid)
            old=c.execute("SELECT canonical FROM events WHERE session=? AND event_id=?", (session,eid)).fetchone()
            can=canonical_json(e)
            if old:
                if old[0]==can:
                    ignored.append(eid); continue
                c.rollback(); return conflict("EVENT_ID_CONFLICT")

            # Wrong revision/node/key/unavailable parent are ignored.
            if e["revision"] != revision:
                ignored.append(eid); continue
            keys,reusable=compute_keys(c,session,inputs,working)
            if e["key"] != keys.get(e["node"]):
                ignored.append(eid); continue
            parent=PARENT[e["node"]]
            if parent is not None and parent not in reusable:
                ignored.append(eid); continue

            tr=transition(working,e["node"],e)
            if tr=="ignore":
                ignored.append(eid); continue
            if tr=="evidence":
                c.rollback(); return conflict("EVIDENCE_CONFLICT")
            if tr=="conflict":
                c.rollback(); return conflict("STATUS_CONFLICT")

            # Accept: reserve the event ID only for accepted events.
            set_state(working,e["node"],e)
            c.execute("INSERT INTO events(session,event_id,revision,canonical) VALUES(?,?,?,?)",
                      (session,eid,revision,can))
            if e["status"]=="succeeded":
                cache_put(c,session,e["node"],e["key"],e["artifactDigest"],eid)
            accepted.append(eid)

        # Persist state only after all events pass conflict checks.
        s["state"]=working
        save_session(c,session,s)
        c.commit()

        # Readback response from the resulting state/cache.
        with sqlite3.connect(DB_PATH) as rc:
            keys,reusable=compute_keys(rc,session,inputs,working)
            nodes=[]
            for node in DAG:
                key=keys[node]
                deps={}
                if key is not None:
                    vals=dependency_values(node,inputs,reusable)
                    for k in NODE_DEPS[node]:
                        deps[k]=vals[k]
                    deps["cacheKey"]=key

                cur=node_state(working,node)
                hit=cache_get(rc,session,node,key) if key is not None else None
                # A successful cache hit is immutable evidence. If it was
                # produced by this/current state, its event triggers reuse.
                if hit:
                    action="reuse"; reason="CACHE_HIT"; trig=[hit["eventId"]]
                elif cur["status"]=="terminal_failed" and cur["key"]==key:
                    action="block"; reason="TERMINAL_FAILURE"; trig=[cur["eventId"]]
                elif cur["status"] in {"started","retryable_failed"} and cur["key"]==key:
                    if cur["status"]=="started":
                        action="block"; reason="RUNNING"
                    else:
                        action="rerun"; reason="RETRYABLE_FAILURE"
                    trig=[cur["eventId"]]
                elif key is None:
                    # Dependency unavailable.
                    parent=PARENT[node]
                    if parent and node_state(working,parent)["status"]=="terminal_failed":
                        action="block"; reason="UPSTREAM_TERMINAL"
                    else:
                        action="block"; reason="UPSTREAM_PENDING"
                    trig=[]
                else:
                    action="rerun"; reason="CACHE_MISS"; trig=[]

                nodes.append({
                    "node":node, "action":action, "reasonCodes":[reason],
                    "dependencyDigests":deps, "triggeringEventIds":trig
                })
        return {"revision":revision,"acceptedEventIds":accepted,
                "ignoredEventIds":ignored,"nodes":nodes}

@app.get("/health")
def health():
    return {"status":"ok"}
