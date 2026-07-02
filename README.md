# Conversational SHL Assessment Recommender

A production-grade, stateless conversational recommender platform designed to assist recruiters and hiring managers in selecting the ideal SHL assessment solutions.

## Architecture Overview

The system processes incoming dialogue history to reconstruct conversation parameters, determine dialogue routing actions, retrieve assessments using a hybrid search algorithm, and format replies conforming strictly to the evaluation schema.

```
                  [ Stateless Request: POST /chat ]
                                 │
                        [ Input Sanitization ]
                                 │
                  [ Dialogue State Tracker (DST) ]
                                 │
                   [ Deterministic Decision Engine ]
                                 │
            ┌────────────────────┴────────────────────┐
            ▼                                         ▼
      [ Action: CLARIFY ]                    [ Action: RECOMMEND ]
            │                                         │
    [ Ask Clarification ]                    [ Hybrid Retrieval ]
            │                                 (BM25 + Semantic)
            │                                         │
            │                              [ Metadata Match & RRF ]
            │                                         │
            │                              [ Cross-Encoder Rerank ]
            │                                         │
            └────────────────────┬────────────────────┘
                                 ▼
                     [ Grounded Response Gen ]
                                 │
                       [ Response Validation ]
                                 │
                     [ Stateless JSON Response ]
```

## Folder Structure

```
├── data/                       # Dataset directories (catalog, traces)
├── src/                        # Main application package
│   ├── agents/                 # Dialogue State Tracking and Decision Engine
│   ├── database/               # Hybrid database & Vector store interfaces
│   ├── models/                 # Pydantic schemas and serialization definitions
│   ├── utils/                  # Structured logging and helper utilities
│   ├── config.py               # Settings and environment variables validator
│   ├── constants.py            # Turn limits and threshold constants
│   └── main.py                 # FastAPI application routes
├── tests/                      # Testing suite (API, Retrieval, and Behavioral probes)
├── Dockerfile                  # Application deployment container configuration
├── pyproject.toml              # Build tool specifications and linters
└── requirements.txt            # System dependencies manifest
```

## Setup & Local Installation

### Prerequisites
* Python 3.12+
* Docker (Optional, for container run)

### Local Virtual Environment
1. Create and activate a Python virtual environment:
   ```bash
   python -m venv myenv
   # On Windows:
   .\myenv\Scripts\activate
   # On macOS/Linux:
   source myenv/bin/activate
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Locally
To launch the FastAPI development server:
```bash
uvicorn src.main:app --reload --port 8000
```
The server will start at `http://127.0.0.1:8000`. You can inspect the health check at `/health`.

## Running Tests
Run the standard test suites using pytest:
```bash
pytest
```
To run tests with code coverage metrics:
```bash
pytest --cov=src
```

## Docker Deployment
1. Build the Docker container image:
   ```bash
   docker build -t shl-assessment-recommender .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 --env-file .env shl-assessment-recommender
   ```
