# Sovereign SQL Engine

Production-focused NL2SQL system with schema-aware retrieval, graph-based join expansion, guarded SQL generation, and optional streaming execution traces.

## Overview

Sovereign SQL Engine converts natural-language analytics questions into executable SQL through a staged pipeline:

1. Safety guardrail validation
2. Query difficulty/topic classification
3. Semantic retrieval of relevant schema metadata
4. Neo4j join-path expansion across table relationships
5. Compact schema assembly for prompt conditioning
6. SQL generation through a hosted inference endpoint
7. Read-only SQL firewall and controlled execution
8. Observability persistence (request records, failures, feedback)

The backend supports both single-shot JSON responses and live Server-Sent Events (SSE) streaming.

## Architecture At Runtime

```text
Client Query
        |
        v
FastAPI Backend (/v1/pipeline/query or /v1/pipeline/stream)
        |
        +-- parallel: Guard + Classifier + Pinecone Retrieval
                                             |
                                             +--> Neo4j join expansion
                                                             |
                                                             +--> Metadata schema builder
                                                                             |
                                                                             +--> RunPod SQL generation
                                                                                             |
                                                                                             +--> SQL firewall (read-only + limit policy)
                                                                                                             |
                                                                                                             +--> SQLite Cloud execution (optional)
                                                                                                                             |
                                                                                                                             +--> API response + observability upsert
```

## Core Capabilities

- Concurrent orchestration with per-stage timeouts
- Safety-first gating (guardrail + SQL firewall)
- Retrieval-augmented schema narrowing (Pinecone + SQLite metadata)
- Join-aware table expansion using Neo4j graph traversal
- OpenAI-compatible model invocation through RunPod/Modal endpoints
- SSE event streaming for stage-by-stage UI updates
- Structured JSON logging with request/trace/span correlation IDs
- In-process latency histograms and readiness/liveness probes
- Durable audit and feedback records in SQLite Cloud

## Repository Structure

```text
sovereign-sql-engine/
├── backend/                  # FastAPI production pipeline API
├── frontend/                 # React + Vite streaming UI
├── pipeline_test/            # CLI pipeline checker (non-API path)
├── metadata_creation/        # Metadata ingestion into SQLite Cloud
├── modal_deployment/         # Modal vLLM OpenAI-compatible deployment
├── model_quantization/       # Quantization/tensorization/publishing utilities
└── vllm_worker/              # RunPod vLLM worker assets (Dockerized)
```

## Technology Stack

- API and orchestration: FastAPI, asyncio, Pydantic
- Vector retrieval and reranking: Pinecone
- Schema graph traversal: Neo4j
- Metadata and query execution store: SQLite Cloud
- Hosted generation endpoints: RunPod + Modal-compatible OpenAI APIs
- Frontend: React, Vite, SSE client parsing
- Deployment assets: Docker (worker), Modal deployment scripts

## Quick Start

### Prerequisites

- Python 3.12+
- Node.js 18+
- uv (Python package manager)
- Access credentials for:
    - SQLite Cloud
    - Pinecone
    - Neo4j
    - RunPod endpoint
    - Guard and classifier model endpoints

### 1) Configure Environment

Create backend env from pipeline template and complete missing fields:

```bash
cp pipeline_test/.env.example backend/.env
```

Then edit `backend/.env` and ensure all required keys exist:

- `DB_NAME`
- `SQLITE_HOST`, `SQLITE_PORT`, `SQLITE_API_KEY`
- `SQLITE_DB`
- `SQLITE_METADATA_DB`
- `SQLITE_LLM_OBSERVABILITY_DB`
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` or `PINECONE_INDEX_HOST`
- `LLAMA_GUARD_URL`
- `PHI4_URL`
- `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`

Notes:

- `backend` reads from `backend/.env`.
- `pipeline_test` reads from its local `.env` when run directly.

### 2) Run Backend

```bash
cd backend
uv sync
PYTHONPATH=.. uv run start
```

Default URL: `http://localhost:8000`

### 3) Run Frontend

```bash
cd frontend
npm install
npm run dev
```

Default URL: `http://localhost:5173`

## API Endpoints

### Pipeline

- `POST /v1/pipeline/query`
    - Runs full pipeline and returns one JSON response.

- `POST /v1/pipeline/stream`
    - Streams stage events via SSE (`pipeline.start`, `guard`, `classification`, `pinecone`, `neo4j`, `schema`, `runpod`, `execution.remark`, `execution.data`, `pipeline.complete`).

- `POST /v1/pipeline/feedback`
    - Records user feedback and logical failure annotations.

### Observability and Health

- `GET /health` (liveness)
- `GET /ready` (dependency readiness)
- `GET /metrics` (in-process counters/histograms)
- `GET /docs` and `GET /redoc` (OpenAPI docs)

## Example Requests

Run full JSON pipeline:

```bash
curl -X POST http://localhost:8000/v1/pipeline/query \
    -H "Content-Type: application/json" \
    -d '{"query": "Which drivers won the most races in 2022?"}'
```

Stream pipeline via SSE:

```bash
curl -N -X POST http://localhost:8000/v1/pipeline/stream \
    -H "Content-Type: application/json" \
    -d '{"query": "List top 10 constructors by total points."}'
```

## Component Notes

### `backend/`

Production FastAPI service with:

- Async pipeline executor and SSE executor
- Dependency-injected service clients
- SQL firewall and execution planning
- Persistent observability records

See: `backend/README.md`

### `frontend/`

Streaming UI for inspecting each pipeline stage, SQL outputs, execution results, and metrics.

See: `frontend/README.md`

### `pipeline_test/`

CLI path to run a single end-to-end query without API server:

```bash
python -m pipeline_test.main "your natural-language query"
```

### `metadata_creation/`

Ingests table/column metadata from CSV descriptions into SQLite Cloud metadata tables used by retrieval and schema assembly.

### `modal_deployment/`

Deploys OpenAI-compatible inference endpoints (for guard/classifier or other model-serving roles) on Modal with GPU-backed vLLM.

### `model_quantization/`

Scripts for quantization workflows, tensorization, model testing, and publishing model artifacts.

### `vllm_worker/`

RunPod-oriented vLLM worker and Docker assets for serverless-compatible model serving.

## Scalability and Reliability Design

- Parallel stage fan-out reduces tail latency
- Per-stage + global timeouts prevent request starvation
- Early exits on blocked/out-of-topic queries reduce compute load
- Externalized model serving (RunPod/Modal) decouples heavy inference from API layer
- Thread-safe service clients used via `asyncio.to_thread`
- Structured telemetry enables root-cause analysis by request and stage

## Current Limitations and Next Steps

- Heavy dependence on external managed services (Pinecone/Neo4j/RunPod)
- End-to-end local development requires complete credential setup
- MLOps lifecycle tooling (experiment registry/evaluation dashboards) is partial and can be expanded

Suggested improvements:

- Add Docker Compose for local backend + frontend + optional mocks
- Add CI for linting/tests and smoke checks
- Add automated SQL AST validation and offline regression suite

## License

See `LICENSE`.