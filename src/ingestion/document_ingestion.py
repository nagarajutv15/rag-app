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

def load_document(
    file_path: str,
):

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

        logger.info(
            "Document Loaded | Pages=%d",
            len(documents),
        )

        return documents

    except HTTPException:

        raise

    except Exception:

        logger.exception(
            "Failed loading document | File=%s",
            file_path,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to load document.",
        )

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Load Document Finished | Time=%.2f ms",
            latency,
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

    try:

        safe_name = os.path.basename(
            file.filename
        )

        unique_name = (
            f"{uuid.uuid4()}_{safe_name}"
        )

        file_path = (
            UPLOAD_DIR / unique_name
        )

        with open(file_path, "wb") as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

        logger.info(
            "Document Saved | Path=%s",
            file_path,
        )

        return str(file_path)

    except Exception:

        logger.exception(
            "Failed saving document | File=%s",
            file.filename,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to save uploaded document.",
        )

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Save Document Finished | Time=%.2f ms",
            latency,
        )


# ----------------------------------------------------------------------------------------------------------
# Chunk Documents
# ----------------------------------------------------------------------------------------------------------

def chunk_documents(
    documents: List[Document],
) -> List[Document]:

    start = time.perf_counter()

    try:

        if not documents:

            logger.warning(
                "No documents available for chunking."
            )

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

        chunks = splitter.split_documents(
            documents
        )

        for chunk in chunks:

            chunk.metadata["chunk_id"] = str(
                uuid.uuid4()
            )

        logger.info(
            "Chunking Completed | Chunks=%d",
            len(chunks),
        )

        return chunks

    except Exception:

        logger.exception(
            "Chunking Failed"
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Chunking Finished | Time=%.2f ms",
            latency,
        )

# ----------------------------------------------------------------------------------------------------------
# Embedding Model
# ----------------------------------------------------------------------------------------------------------




def get_embedding_model():

    global _embedding_model

    try:

        if _embedding_model is None:

            logger.info(
                "Initializing Embedding Model | Model=%s",
                EMBEDDING_MODEL,
            )

            _embedding_model = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                chunk_size=EMBEDDING_BATCH_SIZE,
            )

        return _embedding_model

    except Exception:

        logger.exception(
            "Embedding Model Initialization Failed"
        )

        raise

# ----------------------------------------------------------------------------------------------------------
# Generate Embeddings
# ----------------------------------------------------------------------------------------------------------

def generate_embeddings(
    chunks: List[Document],
):

    start = time.perf_counter()

    try:

        if not chunks:

            logger.warning(
                "No chunks available for embedding generation."
            )

            return []

        logger.info(
            "Generating Embeddings | Chunks=%d",
            len(chunks),
        )

        embedding_model = get_embedding_model()

        texts = [
            chunk.page_content
            for chunk in chunks
        ]

        vectors = embedding_model.embed_documents(
            texts=texts,
        )

        logger.info(
            "Embeddings Generated | Count=%d",
            len(vectors),
        )

        return vectors

    except Exception:

        logger.exception(
            "Embedding Generation Failed"
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Embedding Generation Finished | Time=%.2f ms",
            latency,
        )