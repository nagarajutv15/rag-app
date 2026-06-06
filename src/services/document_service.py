from src.rag.vectorstore.bm25_store import build_bm25_index
from src.models.document_schema import DocumentMetadata
from src.rag.ingestion.document_upload import save_document
from src.rag.ingestion.document_loader import load_document
from src.rag.ingestion.text_cleaner import clean_documents
from src.rag.processing.chunker import chunk_documents
from src.rag.processing.embeddings import generate_embeddings
from src.rag.vectorstore.vector_store import store_vectors
from datetime import datetime


# VERSIONING: When a new document is uploaded with the same name, mark the old one as inactive and increment the version number. 
# This way we can keep track of document history and changes over time.

def process_document_upload(
    department_id,
    source,
    file,
    db
):

    file_path = save_document(
        file
    )

    documents = load_document(
        file_path
    )

    documents = clean_documents(
        documents
    )

    chunks = chunk_documents(
        documents
    )

    # VERSIONING

    existing_document = (

        db.query(DocumentMetadata)

        .filter(
            DocumentMetadata.file_name == file.filename,

            DocumentMetadata.department_id
            == department_id
        )

        .order_by(
            DocumentMetadata.version.desc()
        )

        .first()
    )

    version = 1

    if existing_document:

        existing_document.is_active = False

        db.add(
            existing_document
        )

        version = (
            existing_document.version + 1
        )

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

            "document_id":
                document.document_id,

            "department_id":
                department_id,

            "file_name":
                file.filename,

            "version":
                version,

            "is_active":
                True,

            "uploaded_at":
                uploaded_at.isoformat()
        })

    build_bm25_index(
        chunks
    )

    vectors = generate_embeddings(
        chunks
    )

    store_vectors(
        chunks,
        vectors
    )

    return {

        "message":
            "Document uploaded successfully",

        "document_id":
            document.document_id,

        "department_id":
            department_id,

        "file_name":
            document.file_name,

        "version":
            document.version
    }