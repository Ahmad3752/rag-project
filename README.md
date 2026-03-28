# RAG Project (FastAPI + Streamlit)

Simple Retrieval-Augmented Generation (RAG) application with a FastAPI backend and Streamlit frontend.

Features
- PDF upload and processing
- RAG-based question answering over uploaded documents

Tech stack
- FastAPI
- Streamlit
- LangChain
- FAISS (vector store)

Getting started
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Add environment variables:

```bash
copy .env.example .env
# Edit .env and set OPENROUTER_API_KEY and OPENROUTER_MODEL
```

3. Run the backend (FastAPI / Uvicorn):

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

4. Run the frontend (Streamlit):

```bash
cd frontend
streamlit run app.py
```

Project structure

- backend/
- frontend/
- requirements.txt
- README.md

Suggestions to improve this repo
- Add a `requirements.txt` or `pyproject.toml` at the repo root (already present). Ensure pinned versions for reproducibility.
- Add a short demo GIF or screenshots in `README.md` under a "Demo" heading.
- Add unit tests for critical components and a `tests/` folder.
- Add a `LICENSE` file (MIT is common for small projects).
- Add CI workflow (GitHub Actions) to run linting and tests on push.
- Add a CONTRIBUTING.md and CODE_OF_CONDUCT.md if open-sourcing.

License

This project is provided as-is. Add a license if you plan to publish.
Minimal RAG FastAPI backend

Overview

- The app exposes a minimal FastAPI backend under [backend/main.py](backend/main.py#L1).
- RAG pipeline code lives in [backend/rag_engine.py](backend/rag_engine.py#L1).
- This service expects you to upload a PDF and then ask questions about that PDF; there is no built-in documents file.

Run the app

1. Activate your project virtual environment (example on Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2. Start server:

```bash
uvicorn backend.main:app --reload
```

3. Open Swagger UI:

http://127.0.0.1:8000/docs

API Endpoints

- GET `/` — health check, returns `{ "status": "ok" }`.
- POST `/ask_pdf` — upload a PDF and submit a `question` form field (multipart/form-data). Returns `{ "answer": "..." }`.

How it works (code explanation)

- Initialization (`init_rag()`): located in [backend/rag_engine.py](backend/rag_engine.py#L1).
  - Purpose: initialize the reusable pieces of the pipeline (the prompt template and the LLM client).
  - Why we do this at startup: creating the LLM client and compiling the prompt once saves time; we avoid re-initializing the LLM per request which would be slow and wasteful.

- Per-request PDF handling (`get_rag_response_from_pdf`): in [backend/rag_engine.py](backend/rag_engine.py#L1).
  - Flow: the uploaded PDF is saved temporarily, loaded with `PyPDFLoader`, split into chunks, embeddings are created, and an in-memory FAISS index is built.
  - We then create a temporary chain that uses the already-initialized `prompt` and `llm` to answer the query.
  - The temporary index is discarded after the request and the upload file is removed. This keeps the app stateless and simple.

- Why not persist indexes: persisting (two-step: upload -> store index -> query later) requires storage management and bookkeeping. Per-request indexing is simpler for minimal setups.

Key files

- [backend/rag_engine.py](backend/rag_engine.py#L1): RAG logic — `init_rag()`, `get_rag_response_from_pdf()`.
- [backend/main.py](backend/main.py#L1): FastAPI app — startup event calls `init_rag()` and exposes `/ask_pdf`.

Environment variables (.env)

- `OPENROUTER_API_KEY` (preferred) or `OPENAI_API_KEY` (fallback) — your LLM service key.
- `OPENROUTER_BASE_URL` — base URL for OpenRouter-compatible endpoints (default used in code).
- `OPENROUTER_MODEL` — model identifier (used by `ChatOpenAI`).
- `RAG_QUIET=1` (default) — suppresses noisy model-download logs during embedding creation. Set to `0` to show logs.

Changing the model

- Edit the `.env` file and set `OPENROUTER_MODEL` to the model you want. The code reads this value at startup when creating the LLM client ([backend/rag_engine.py](backend/rag_engine.py#L69-L73)).

Requirements

- `requirements.txt` contains the (unfrozen) main packages used by this project. Install with `python -m pip install -r requirements.txt`.

Security

- Never commit your API keys. If a key was committed, rotate or revoke it immediately.

If something breaks

- Paste the exact traceback or the HTTP response body you see and I will debug it.

