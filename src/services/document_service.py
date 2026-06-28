import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from src.ingestion.document_ingestion import (
    load_document,
    save_document,
    chunk_documents,
    generate_embeddings,
)
from src.models.document_schema import DocumentMetadata
from src.utils.logger import logger
from src.vectorstore.bm25_store import (
    build_bm25_index,
    remove_document_chunks,
)
from src.vectorstore.vector_store import (
    delete_document_vectors,
    store_vectors,
)


# ----------------------------------------------------------------------------------------------------------
# Process Document Upload
# ----------------------------------------------------------------------------------------------------------

def process_document_upload(
    source,
    file,
    db,
):

    start = time.perf_counter()

    logger.info(
        "Processing Document Upload | File=%s",
        file.filename,
    )

    try:

        file_path = save_document(file)

        documents = load_document(file_path)

        chunks = chunk_documents(documents)

        # ------------------------------------------------------------------
        # Versioning
        # ------------------------------------------------------------------

        existing_document = (
            db.query(DocumentMetadata)
            .filter(
                DocumentMetadata.file_name == file.filename,
            )
            .order_by(
                DocumentMetadata.version.desc()
            )
            .first()
        )

        version = 1

        if existing_document:

            logger.info(
                "Existing Document Found | Version=%d",
                existing_document.version,
            )

            remove_document_chunks(
                existing_document.document_id
            )

            delete_document_vectors(
                existing_document.document_id
            )

            existing_document.is_active = False

            db.add(existing_document)

            version = existing_document.version + 1

        uploaded_at = datetime.now(
            timezone.utc
        )

        document = DocumentMetadata(
            file_name=file.filename,
            source=source,
            version=version,
            file_path=file_path,
            is_active=True,
            uploaded_at=uploaded_at,
        )

        # ------------------------------------------------------------------
        # Save Metadata
        # ------------------------------------------------------------------

        try:

            db.add(document)

            db.commit()

            db.refresh(document)

            logger.info(
                "Document Metadata Saved | DocumentId=%s",
                document.document_id,
            )

        except Exception:

            db.rollback()

            logger.exception(
                "Failed To Save Document Metadata"
            )

            raise

        # ------------------------------------------------------------------
        # Chunk Metadata
        # ------------------------------------------------------------------

        for chunk in chunks:

            chunk.metadata.update(
                {
                    "document_id": document.document_id,
                    "file_name": file.filename,
                    "version": version,
                    "is_active": True,
                    "uploaded_at": uploaded_at.isoformat(),
                }
            )

        # ------------------------------------------------------------------
        # BM25 + Embeddings
        # ------------------------------------------------------------------

        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:

            bm25_future = executor.submit(
                build_bm25_index,
                chunks,
            )

            embedding_future = executor.submit(
                generate_embeddings,
                chunks,
            )

            bm25_future.result()

            vectors = embedding_future.result()

        logger.info(
            "BM25 Index And Embeddings Generated"
        )

        # ------------------------------------------------------------------
        # Store Vectors
        # ------------------------------------------------------------------

        store_vectors(
            chunks,
            vectors,
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Document Upload Completed | File=%s | Time=%.2f ms",
            file.filename,
            latency,
        )

        return {

            "message": "Document uploaded successfully",

            "document_id": document.document_id,

            "file_name": document.file_name,

            "version": document.version,

        }

    except Exception:

        logger.exception(
            "Document Upload Processing Failed | File=%s",
            file.filename,
        )

        raise

