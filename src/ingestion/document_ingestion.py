import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils.logger import logger


load_dotenv()


# ---------------------------------------------------------
# Upload Directory
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

EMBEDDING_BATCH_SIZE = int(
    os.getenv(
        "EMBEDDING_BATCH_SIZE",
        "100",
    )
)

_embedding_model = None


# ----------------------------------------------------------------------------------------------------------
# Load Document
# ----------------------------------------------------------------------------------------------------------

def load_document(file_path: str):

    start = time.perf_counter()

    logger.info(
        "Loading Document | File=%s",
        file_path,
    )

    try:

        if file_path.endswith(".pdf"):

            loader = PyPDFLoader(file_path)

        elif file_path.endswith(".txt"):

            loader = TextLoader(
                file_path,
                encoding="utf-8",
            )

        elif file_path.endswith(".docx"):

            loader = Docx2txtLoader(file_path)

        else:

            raise HTTPException(
                status_code=400,
                detail="Only PDF, TXT and DOCX files are supported.",
            )

        documents = loader.load()

        if not documents:

            raise HTTPException(
                status_code=400,
                detail="No content found in document.",
            )

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Document Loaded | Pages=%d | Time=%.2f ms",
            len(documents),
            latency,
        )

        return documents

    except HTTPException:

        raise

    except Exception as e:

        logger.exception(
            "Document Loading Failed | File=%s",
            file_path,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Document loading failed: {str(e)}",
        )


# ----------------------------------------------------------------------------------------------------------
# Save Document
# ----------------------------------------------------------------------------------------------------------

def save_document(
    file: UploadFile,
):

    start = time.perf_counter()

    logger.info(
        "Saving Document | File=%s",
        file.filename,
    )

    safe_name = os.path.basename(file.filename)

    unique_name = f"{uuid.uuid4()}_{safe_name}"

    file_path = UPLOAD_DIR / unique_name

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer,
        )

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Document Saved | Path=%s | Time=%.2f ms",
        file_path,
        latency,
    )

    return str(file_path)


# ----------------------------------------------------------------------------------------------------------
# Chunk Documents
# ----------------------------------------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
) -> List[Document]:

    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(documents)

    for chunk in chunks:

        chunk.metadata["chunk_id"] = str(
            uuid.uuid4()
        )

    return chunks


# ----------------------------------------------------------------------------------------------------------
# Embedding Model
# ----------------------------------------------------------------------------------------------------------




def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:

        _embedding_model = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            chunk_size=EMBEDDING_BATCH_SIZE,
        )

    return _embedding_model


# ----------------------------------------------------------------------------------------------------------
# Generate Embeddings
# ----------------------------------------------------------------------------------------------------------

def generate_embeddings(
    chunks: List[Document],
):

    if not chunks:
        return []

    start = time.perf_counter()

    logger.info(
        "Generating Embeddings | Chunks=%d",
        len(chunks),
    )

    embeddings_model = get_embedding_model()

    texts = [
        chunk.page_content
        for chunk in chunks
    ]

    vectors = embeddings_model.embed_documents(
        texts=texts,
    )

    latency = (time.perf_counter() - start) * 1000

    logger.info(
        "Embeddings Generated | Vectors=%d | Time=%.2f ms",
        len(vectors),
        latency,
    )

    return vectors