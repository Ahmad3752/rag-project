"""Simple, clear RAG helpers used by the FastAPI app.

Design goals:
- Initialize the LLM and prompt once at startup (`init_rag`).
- Index uploaded PDFs in-memory per request (`get_rag_response_from_pdf`).
"""
import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout, nullcontext

from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Reduce noisy HF logs by default; keep adjustable via env
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore")

# Module-level shared objects (set by init_rag)
llm = None
prompt = None

QUIET_SETUP = os.getenv("RAG_QUIET", "1") == "1"


def _quiet_context():
    if QUIET_SETUP:
        sink = io.StringIO()
        return redirect_stdout(sink), redirect_stderr(sink)
    return nullcontext(), nullcontext()


def init_rag():
    """Initialize reusable components (LLM client + prompt). Idempotent."""
    global llm, prompt
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY in .env")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template(
        """
Answer the question based only on the context below.

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model_name, temperature=0)


def get_rag_response_from_pdf(file_path: str, question: str) -> str:
    """Index the provided PDF in-memory and return an answer for `question`.

    Steps:
    1. Load PDF pages
    2. Split into chunks
    3. Create embeddings and an in-memory FAISS index
    4. Retrieve top chunks and ask the LLM
    """
    # ensure core pieces exist
    if prompt is None or llm is None:
        init_rag()

    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(pages)

    out_ctx, err_ctx = _quiet_context()
    with out_ctx, err_ctx:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vs = FAISS.from_documents(docs, embeddings)

    # `as_retriever()` returns a VectorStoreRetriever which may not expose
    # `get_relevant_documents()` in all langchain versions. Use the FAISS
    # API directly for compatibility.
    top_docs = vs.similarity_search(question, k=3)
    context_text = "\n\n".join(d.page_content for d in top_docs)

    final_prompt = (
        "Answer the question based only on the context below.\n\n"
        f"Context:\n{context_text}\n\nQuestion:\n{question}\n\nAnswer:\n"
    )

    # Use the initialized LLM to get a response. Call llm.invoke with the
    # composed prompt string and return the content.
    res = llm.invoke(final_prompt)
    return getattr(res, "content", str(res))


def get_rag_response(_query: str) -> str:
    """This app does not keep a default index; ask via `/ask_pdf`."""
    raise RuntimeError("No documents indexed. Upload a PDF via /ask_pdf and retry.")
