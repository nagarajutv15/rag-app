import time
from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

from src.utils.logger import logger
from src.vectorstore.qdrant_connection import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION,
)


# ----------------------------------------------------------------------------------------------------------
# Store Vectors
# ----------------------------------------------------------------------------------------------------------

def store_vectors(
    chunks: List[Document],
    vectors: List[list],
):

    start = time.perf_counter()

    try:

        if not chunks or not vectors:

            logger.warning(
                "No vectors available to store."
            )

            return

        logger.info(
            "Vector Store Started | Chunks=%d | Vectors=%d",
            len(chunks),
            len(vectors),
        )

        points = []

        for chunk, vector in zip(chunks, vectors):

            payload = build_payload(chunk)

            points.append(

                PointStruct(

                    id=str(uuid4()),

                    vector=vector,

                    payload=payload,

                )

            )

        QDRANT_CLIENT.upsert(

            collection_name=QDRANT_COLLECTION,

            points=points,

        )

        logger.info(
            "Vector Store Successful | Stored=%d",
            len(points),
        )

    except Exception:

        logger.exception(
            "Failed storing vectors."
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Vector Store Finished | Time=%.2f ms",
            latency,
        )


# ----------------------------------------------------------------------------------------------------------
# Build Payload
# ----------------------------------------------------------------------------------------------------------

def build_payload(
    chunk: Document,
):

    try:

        metadata = chunk.metadata

        return {

            "chunk_id": metadata.get("chunk_id"),

            "document_id": metadata.get("document_id"),

            "file_name": metadata.get("file_name"),

            "version": metadata.get(
                "version",
                1,
            ),

            "is_active": metadata.get(
                "is_active",
                True,
            ),

            "text": chunk.page_content,

        }

    except Exception:

        logger.exception(
            "Failed building Qdrant payload."
        )

        raise


# ----------------------------------------------------------------------------------------------------------
# Delete Document Vectors
# ----------------------------------------------------------------------------------------------------------

def delete_document_vectors(
    document_id: int,
):

    start = time.perf_counter()

    try:

        logger.info(
            "Deleting Document Vectors | Document=%s",
            document_id,
        )

        QDRANT_CLIENT.delete(

            collection_name=QDRANT_COLLECTION,

            points_selector=Filter(

                must=[

                    FieldCondition(

                        key="document_id",

                        match=MatchValue(
                            value=document_id,
                        ),

                    )

                ]

            ),

        )

        logger.info(
            "Document Vectors Deleted | Document=%s",
            document_id,
        )

    except Exception:

        logger.exception(
            "Failed deleting vectors | Document=%s",
            document_id,
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Delete Vector Operation Finished | Time=%.2f ms",
            latency,
        )