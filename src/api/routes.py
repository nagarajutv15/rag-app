from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    Form
)

from sqlalchemy.orm import Session

from src.models.database import get_db
from src.models.document_schema import DocumentMetadata

from src.rag.ingestion.document_upload import save_document
from src.rag.ingestion.document_loader import load_document
from src.rag.ingestion.text_cleaner import clean_documents
from src.rag.processing.chunker import chunk_documents
from src.rag.processing.embeddings import generate_embeddings
from src.rag.vectorstore.vector_store import store_vectors


router = APIRouter()


@router.post("/documents/upload")
def upload_document(
    source: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    file_path = save_document(file)

    documents = load_document(file_path)

    documents = clean_documents(documents)

    chunks = chunk_documents(documents)

    vectors = generate_embeddings(chunks)

    store_vectors(
        chunks,
        vectors
    )

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

        existing_document.is_active = False

        db.add(existing_document)

        version = existing_document.version + 1


    document = DocumentMetadata(

        file_name=file.filename,
        source=source,
        version=version,
        file_path=file_path,
        is_active=True
    )


    try:

        db.add(document)

        db.commit()

        db.refresh(document)

    except Exception:

        db.rollback()

        raise


    return {

        "message":"Document uploaded successfully",

        "document_id":document.document_id,

        "file_name":document.file_name,

        "version":document.version
    }