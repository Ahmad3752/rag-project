"""Minimal FastAPI app exposing the RAG engine.

Run with:
    uvicorn backend.main:app --reload

Docs: /docs
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import os
from pydantic import BaseModel

from . import rag_engine

app = FastAPI(title="Minimal RAG API")


class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    # Initialize RAG pipeline once
    rag_engine.init_rag()


@app.get("/", response_model=dict)
def root():
    return {"status": "ok"}


@app.post("/ask_pdf", response_model=AnswerResponse)
async def ask_pdf(file: UploadFile = File(...), question: str = Form(...)):
    """Upload a PDF and ask a question in the same request.

    This keeps the flow simple: the PDF is indexed in-memory per-request.
    """
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file.filename)
    try:
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        answer = rag_engine.get_rag_response_from_pdf(file_path, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            os.remove(file_path)
        except Exception:
            pass
