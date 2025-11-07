# modal deploy server.py  # prod

import os
from datetime import datetime, timezone

import modal
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from pymongo import MongoClient, UpdateOne

image = modal.Image.debian_slim().pip_install(
    "fastapi[standard]", "pydantic", "pymongo>=4.7"
)

app = modal.App("llm-annotation-endpoint", image=image)

secrets = modal.Secret.from_name("mongodb-secret-corral")


class NodeAnnotation(BaseModel):
    markers: list[str] = Field(default_factory=list)
    notes: str = ""


class FileNode(BaseModel):
    id: str
    type: str
    annotatable: bool


class Payload(BaseModel):
    annotator: str  # will be ignored in favor of token identity
    mongodbKey: str
    annotations: dict[str, dict[str, NodeAnnotation]]
    traceComments: dict[str, str] = Field(default_factory=dict)
    fileNodes: dict[str, list[FileNode]] = Field(default_factory=dict)

    @validator("mongodbKey")
    def non_empty_key(cls, v: str):
        if not v or not v.strip():
            raise ValueError("mongodbKey must be a non-empty string")
        return v.strip()


def is_key_allowed(mongodb_key: str) -> bool:
    """Check if the mongodb_key is in the allowed list."""
    allowed = {
        k.strip()
        for k in os.environ.get("ALLOWED_MONGODB_KEYS", "").split(",")
        if k.strip()
    }
    return not (allowed and mongodb_key not in allowed)


def _build_fastapi_app() -> FastAPI:
    web = FastAPI(title="LLM Trace Annotation Ingest", version="1.1.0")

    # CORS - allow all origins for the annotation tool
    # Note: ALLOW_ORIGINS should be comma-separated origins (scheme + host + port ONLY, no paths!)
    # Example: "https://lamalab-org.github.io,http://localhost,http://127.0.0.1"
    # NOT: "https://lamalab-org.github.io/corral/trace-visualizer/" (paths are not part of Origin header)
    allow_origins_env = os.environ.get("ALLOW_ORIGINS", "")

    if allow_origins_env and allow_origins_env != "*":
        # Specific origins provided
        allow_origins = [o.strip() for o in allow_origins_env.split(",")]
        web.add_middleware(
            CORSMiddleware,
            allow_origins=allow_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        # Allow all origins - credentials disabled to allow wildcard
        web.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    mongo_client: MongoClient | None = None

    def _get_client() -> MongoClient:
        nonlocal mongo_client
        if mongo_client is None:
            uri = os.environ.get("MONGODB_URI")
            if not uri:
                raise RuntimeError(
                    "MONGODB_URI is not set. Provide it via Modal Secret."
                )
            # SRV URIs default to TLS; additional TLS options configurable if needed.
            mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        return mongo_client

    def _get_collection():
        db_name = os.environ.get("MONGODB_DB", "Corral")
        coll_name = os.environ.get("MONGODB_COLLECTION", "First-traces")
        return _get_client()[db_name][coll_name]

    @web.options("/ingest")
    def ingest_options():
        """Handle CORS preflight requests."""
        return Response(status_code=200)

    @web.post("/ingest")
    def ingest(payload: Payload):
        # 1) AuthZ: check mongodbKey against allow-list
        if not is_key_allowed(payload.mongodbKey):
            raise HTTPException(status_code=403, detail="mongodbKey is not allowed")

        # 2) Write
        coll = _get_collection()
        now = datetime.now(tz=timezone.utc)
        ops: list[UpdateOne] = []

        for file_id, nodes_dict in payload.annotations.items():
            file_comment = payload.traceComments.get(file_id)
            file_nodes = payload.fileNodes.get(file_id, [])
            node_annotations = {nid: ann.dict() for nid, ann in nodes_dict.items()}

            doc = {
                "mongodbKey": payload.mongodbKey,
                "fileId": file_id,
                "annotator": payload.annotator,
                "annotations": node_annotations,
                "traceComment": file_comment,
                "fileNodes": [fn.dict() for fn in file_nodes],
                "updatedAt": now,
            }

            ops.append(
                UpdateOne(
                    {"mongodbKey": payload.mongodbKey, "fileId": file_id},
                    {"$set": doc, "$setOnInsert": {"createdAt": now}},
                    upsert=True,
                )
            )

        if not ops:
            raise HTTPException(status_code=400, detail="No annotations provided")

        result = coll.bulk_write(ops, ordered=False)
        return {
            "ok": True,
            "matched": result.matched_count,
            "upserted": len(result.upserted_ids or {}),
            "modified": result.modified_count,
        }

    @web.get("/healthz")
    def healthz():
        try:
            _get_client().admin.command("ping")
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    return web


@app.function(secrets=[secrets])
@modal.asgi_app()
def fastapi_app():
    return _build_fastapi_app()
