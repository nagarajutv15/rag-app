from datetime import datetime
from src.models.document_schema import DocumentMetadata
from src.rag.ingestion.document_ingestion import (
    load_document, 
    save_document, 
    chunk_documents, 
    generate_embeddings
)
from src.rag.vectorstore.bm25_store import build_bm25_index
from src.rag.vectorstore.vector_store import store_vectors




#----------------------------------------------------------------------------------------------------------#
# This function processes the uploaded document, saves it to the database, and generates embeddings for the document chunks.

def process_document_upload(
    department_id,
    source,
    file,
    db
):

    file_path = save_document(file, department_id=department_id)
    documents = load_document(file_path)
    chunks = chunk_documents(documents)

    # VERSIONING
    existing_document = (
        db.query(DocumentMetadata)
        .filter(
            DocumentMetadata.file_name == file.filename,
            DocumentMetadata.department_id == department_id
        )
        .order_by(DocumentMetadata.version.desc())
        .first()
    )

    version = 1

    if existing_document:
        existing_document.is_active = False
        db.add(existing_document)
        version = existing_document.version + 1

    uploaded_at = datetime.utcnow()

    document = DocumentMetadata(
        file_name=file.filename,
        department_id=department_id,
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
            "department_id": department_id,
            "file_name": file.filename,
            "version": version,
            "is_active": True,
            "uploaded_at": uploaded_at.isoformat()
        })

    # Build BM25 index and generate embeddings for the document chunks
    build_bm25_index(chunks)

    vectors = generate_embeddings(chunks)

    store_vectors(chunks, vectors)

    return {
        "message": "Document uploaded successfully",
        "document_id": document.document_id,
        "department_id": department_id,
        "file_name": document.file_name,
        "version": document.version
    }