from urllib import response
from src.rag.vectorstore.bm25_store import build_bm25_index
from sqlalchemy.orm import Session
from unstructured import documents
from src.models.database import get_db
from src.models.document_schema import DocumentMetadata
from src.rag.ingestion.document_upload import save_document
from src.rag.ingestion.document_loader import load_document
from src.rag.ingestion.text_cleaner import clean_documents
from src.rag.processing.chunker import chunk_documents
from src.rag.processing.embeddings import generate_embeddings
from src.rag.processing.metadata_extractor import extract_metadata
from src.rag.retrieval.hybrid_search import hybrid_search
from src.rag.vectorstore.vector_store import store_vectors
from src.rag.prompts.templates import build_prompt
from src.llm.llm_service import generate_response
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    Form
)

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

    full_text = "\n".join(
        [doc.page_content for doc in documents]
    )

    document_metadata = extract_metadata(
        full_text
    )
    chunks = chunk_documents(documents)

    for chunk in chunks:

        chunk.metadata.update(
            document_metadata
        )

    build_bm25_index(chunks)

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




@router.get("/ask")
def ask_question(query: str):

    retrieved_docs = hybrid_search(
        query=query,
        top_k=3
    )

    if not retrieved_docs:

        return {
            "question": query,
            "answer": "I couldn't find information in the documents."
        }

    context_chunks = []

    for doc in retrieved_docs:

        chunk_context = f"""
        Text:
        {doc.get("text", "")}

        Emails:
        {", ".join(doc.get("emails", []))}

        Phone Numbers:
        {", ".join(doc.get("phone_numbers", []))}
        """

        context_chunks.append(chunk_context)

    prompt = build_prompt(
        question=query,
        context_chunks=context_chunks
    )

    answer = generate_response(prompt)

    return {
        "question": query,
        "answer": answer,
        "sources": [
            doc["chunk_id"]
            for doc in retrieved_docs
        ]
    }
