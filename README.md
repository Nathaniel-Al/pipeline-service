# Stateful Pipeline Controller

FastAPI service implementing `POST /pipeline`.

For Render, deploy as a Docker Web Service. A persistent disk mounted at `/data` is recommended so SQLite state survives restarts/redeploys. The service uses `/data/pipeline.db` by default.
