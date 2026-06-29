import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

from sqlalchemy.exc import SQLAlchemyError

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


# --------------------------------------------------------------------------------------------------
# Process Document Upload
# --------------------------------------------------------------------------------------------------

def process_document_upload(
    source,
    file,
    db,
):

    start = time.perf_counter()

    logger.info(
        "Document Upload Started | File=%s | Source=%s",
        file.filename,
        source,
    )

    try:

        # ------------------------------------------------------------------
        # Save Uploaded File
        # ------------------------------------------------------------------

        file_path = save_document(file)

        logger.info(
            "File Saved | Path=%s",
            file_path,
        )

        # ------------------------------------------------------------------
        # Load Document
        # ------------------------------------------------------------------

        documents = load_document(file_path)

        if not documents:

            raise ValueError(
                "No content could be extracted from the uploaded document."
            )

        logger.info(
            "Document Loaded | Pages=%d",
            len(documents),
        )

        # ------------------------------------------------------------------
        # Chunk Document
        # ------------------------------------------------------------------

        chunks = chunk_documents(documents)

        if not chunks:

            raise ValueError(
                "No chunks were generated from the document."
            )

        logger.info(
            "Chunking Completed | Chunks=%d",
            len(chunks),
        )

        # ------------------------------------------------------------------
        # Versioning
        # ------------------------------------------------------------------

        existing_document = (

            db.query(DocumentMetadata)

            .filter(
                DocumentMetadata.file_name == file.filename
            )

            .order_by(
                DocumentMetadata.version.desc()
            )

            .first()

        )

        version = 1

        if existing_document:

            logger.info(
                "Existing Document Found | Document=%s | Version=%d",
                existing_document.document_id,
                existing_document.version,
            )

            remove_document_chunks(
                existing_document.document_id,
            )

            delete_document_vectors(
                existing_document.document_id,
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

        db.add(document)

        db.commit()

        db.refresh(document)

        logger.info(
            "Metadata Saved | Document=%s | Version=%d",
            document.document_id,
            document.version,
        )

        # ------------------------------------------------------------------
        # Update Chunk Metadata
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

        logger.info(
            "Chunk Metadata Updated | Chunks=%d",
            len(chunks),
        )

                # ------------------------------------------------------------------
        # Build BM25 + Generate Embeddings
        # ------------------------------------------------------------------

        with ThreadPoolExecutor(max_workers=2) as executor:

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
            "BM25 Index Built & Embeddings Generated | Vectors=%d",
            len(vectors),
        )

        # ------------------------------------------------------------------
        # Store Vectors
        # ------------------------------------------------------------------

        store_vectors(
            chunks=chunks,
            vectors=vectors,
        )

        logger.info(
            "Vectors Stored Successfully | Count=%d",
            len(vectors),
        )

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Document Upload Completed | File=%s | Version=%d | Chunks=%d | Time=%.2f ms",
            file.filename,
            version,
            len(chunks),
            latency,
        )

        return {

            "message": "Document uploaded successfully.",

            "document_id": document.document_id,

            "file_name": document.file_name,

            "version": document.version,

        }

    except SQLAlchemyError:

        db.rollback()

        logger.exception(
            "Database Error During Document Upload | File=%s",
            file.filename,
        )

        raise

    except ValueError:

        db.rollback()

        logger.exception(
            "Document Validation Failed | File=%s",
            file.filename,
        )

        raise

    except Exception:

        db.rollback()

        logger.exception(
            "Unexpected Error During Document Upload | File=%s",
            file.filename,
        )

        raise

    finally:

        latency = (
            time.perf_counter() - start
        ) * 1000

        logger.info(
            "Document Upload Finished | File=%s | Total Time=%.2f ms",
            file.filename,
            latency,
        )