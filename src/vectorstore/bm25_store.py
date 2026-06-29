import time

from rank_bm25 import BM25Okapi

from src.models.document_schema import DocumentMetadata
from src.ingestion.document_ingestion import (
    load_document,
    chunk_documents,
)
from src.utils.logger import logger


BM25_INDEX = None
BM25_DOCUMENTS = []


# ----------------------------------------------------------------------------------------------------------
# Build BM25 Index
# ----------------------------------------------------------------------------------------------------------

def build_bm25_index(chunks):

    start = time.perf_counter()

    global BM25_INDEX
    global BM25_DOCUMENTS

    try:

        logger.info(
            "Building BM25 Index | New Chunks=%d",
            len(chunks),
        )

        BM25_DOCUMENTS.extend(chunks)

        tokenized_docs = [
            chunk.page_content.split()
            for chunk in BM25_DOCUMENTS
        ]

        BM25_INDEX = BM25Okapi(tokenized_docs)

    except Exception:

        logger.exception(
            "Failed to build BM25 index."
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "BM25 Build Finished | Total Chunks=%d | Time=%.2f ms",
            len(BM25_DOCUMENTS),
            latency,
        )

# ----------------------------------------------------------------------------------------------------------
# BM25 Search
# ----------------------------------------------------------------------------------------------------------

def bm25_search(
    query: str,
    top_k: int = 10,
):

    start = time.perf_counter()

    global BM25_INDEX
    global BM25_DOCUMENTS

    logger.info(
        "BM25 Search Started | Query=%s",
        query,
    )

    try:

        if BM25_INDEX is None:

            logger.warning(
                "BM25 Index Not Initialized."
            )

            return []

        tokenized_query = query.split()

        scores = BM25_INDEX.get_scores(
            tokenized_query
        )

        scored_docs = []

        for chunk, score in zip(
            BM25_DOCUMENTS,
            scores,
        ):

            scored_docs.append(
                (chunk, score)
            )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for chunk, score in scored_docs[:top_k]:

            results.append(
                {
                    "chunk_id": chunk.metadata.get("chunk_id"),
                    "document_id": chunk.metadata.get("document_id"),
                    "file_name": chunk.metadata.get("file_name"),
                    "version": chunk.metadata.get("version"),
                    "is_active": chunk.metadata.get("is_active"),
                    "text": chunk.page_content,
                    "bm25_score": float(score),
                }
            )

        logger.info(
            "BM25 Search Completed | Results=%d",
            len(results),
        )

        return results

    except Exception:

        logger.exception(
            "BM25 Search Failed | Query=%s",
            query,
        )

        return []

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "BM25 Search Finished | Time=%.2f ms",
            latency,
        )

# ----------------------------------------------------------------------------------------------------------
# Rebuild BM25 Index
# ----------------------------------------------------------------------------------------------------------

def rebuild_bm25_index(db):

    start = time.perf_counter()

    global BM25_DOCUMENTS
    global BM25_INDEX

    logger.info(
        "Rebuilding BM25 Index"
    )

    try:

        BM25_DOCUMENTS = []
        BM25_INDEX = None

        active_documents = (

            db.query(DocumentMetadata)

            .filter(
                DocumentMetadata.is_active == True
            )

            .all()

        )

        all_chunks = []

        for document in active_documents:

            try:

                documents = load_document(
                    document.file_path
                )

                chunks = chunk_documents(
                    documents
                )

                for chunk in chunks:

                    chunk.metadata.update(
                        {
                            "document_id": document.document_id,
                            "file_name": document.file_name,
                            "version": document.version,
                            "is_active": document.is_active,
                        }
                    )

                all_chunks.extend(chunks)

            except Exception:

                logger.exception(
                    "Failed Loading Document | File=%s",
                    document.file_name,
                )

        if all_chunks:

            build_bm25_index(
                all_chunks
            )

    except Exception:

        logger.exception(
            "BM25 Rebuild Failed."
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "BM25 Rebuild Finished | Chunks=%d | Time=%.2f ms",
            len(BM25_DOCUMENTS),
            latency,
        )


# ----------------------------------------------------------------------------------------------------------
# Remove Document Chunks
# ----------------------------------------------------------------------------------------------------------

def remove_document_chunks(
    document_id: int,
):

    start = time.perf_counter()

    global BM25_DOCUMENTS
    global BM25_INDEX

    logger.info(
        "Removing BM25 Chunks | Document=%s",
        document_id,
    )

    try:

        BM25_DOCUMENTS = [

            chunk

            for chunk in BM25_DOCUMENTS

            if chunk.metadata.get("document_id") != document_id

        ]

        if BM25_DOCUMENTS:

            tokenized_docs = [

                chunk.page_content.split()

                for chunk in BM25_DOCUMENTS

            ]

            BM25_INDEX = BM25Okapi(
                tokenized_docs
            )

        else:

            BM25_INDEX = None

    except Exception:

        logger.exception(
            "Failed Removing BM25 Chunks | Document=%s",
            document_id,
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "BM25 Chunks Removed | Remaining=%d | Time=%.2f ms",
            len(BM25_DOCUMENTS),
            latency,
        )