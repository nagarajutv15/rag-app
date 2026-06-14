from sqlalchemy.orm import Session

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    Form
)

from src.models.database import get_db

from src.rag.adaptive.query_classifier import classify_query
from src.services.document_service import (
    process_document_upload
)

from src.services.chat_service import (
    ask_rag_question
)

router = APIRouter()


@router.post("/documents/upload")
def upload_document(
    department_id: str = Form(...),
    source: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    return process_document_upload(
        department_id=department_id,
        source=source,
        file=file,
        db=db
    )


@router.post("/ask")
def ask_question(
    session_id: str = Form(...),
    query: str = Form(...),
    db: Session = Depends(get_db)
):

    return ask_rag_question(
        query=query,
        session_id=session_id,
        db=db
    )


@router.get("/route")
def test_route(
    query: str
):

    route = classify_query(
        query
    )

    return {
        "route": route
    }