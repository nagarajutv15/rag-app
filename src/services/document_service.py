from datetime import datetime, timezone
from src.models.document_schema import DocumentMetadata
from src.ingestion.document_ingestion import (
    load_document, 
    save_document, 
    chunk_documents, 
    generate_embeddings
)
from src.vectorstore.bm25_store import build_bm25_index
from src.vectorstore.vector_store import store_vectors
from src.vectorstore.bm25_store import remove_document_chunks
from src.vectorstore.vector_store import delete_document_vectors
from concurrent.futures import ThreadPoolExecutor



#----------------------------------------------------------------------------------------------------------#
# This function processes the uploaded document, saves it to the database, and generates embeddings for the document chunks.

def process_document_upload(
    source,
    file,
    db
):

    file_path = save_document(file)
    documents = load_document(file_path)
    chunks = chunk_documents(documents)

    # VERSIONING
    existing_document = (
        db.query(DocumentMetadata)
        .filter(
            DocumentMetadata.file_name == file.filename,
        )
        .order_by(DocumentMetadata.version.desc())
        .first()
    )

    version = 1

    if existing_document:

        remove_document_chunks(existing_document.document_id)
        delete_document_vectors(existing_document.document_id)
        existing_document.is_active = False
        db.add(existing_document)
        version = existing_document.version + 1


    uploaded_at = datetime.now(timezone.utc)

    document = DocumentMetadata(
        file_name=file.filename,
        source=source,
        version=version,
        file_path=file_path,
        is_active=True,
        uploaded_at=uploaded_at
    )


    # Save the document metadata to the database with error handling
    try:
        db.add(document)
        db.commit()
        db.refresh(document)

    except Exception:
        db.rollback()
        raise


    # CHUNK METADATA
    for chunk in chunks:
        chunk.metadata.update({
            "document_id": document.document_id,
            "file_name": file.filename,
            "version": version,
            "is_active": True,
            "uploaded_at": uploaded_at.isoformat()
        })


    # Build BM25 index and generate embeddings for the document chunks
    with ThreadPoolExecutor(max_workers=2) as executor:

        bm25_future = executor.submit(build_bm25_index, chunks)
        embedding_future = executor.submit(generate_embeddings, chunks)
        bm25_future.result()
        vectors = embedding_future.result()


    store_vectors(chunks, vectors)

    return {
        "message": "Document uploaded successfully",
        "document_id": document.document_id,
        "file_name": document.file_name,
        "version": document.version
    }



#----------------------------------------------------------------------------------------------------------#


