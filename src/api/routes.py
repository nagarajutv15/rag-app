from sqlalchemy.orm import Session
from src.models.database import get_db
from src.services.document_service import process_document_upload
from src.services.memory_service import create_session
from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    Form
)


router = APIRouter()

#----------------------------------------------------------------------------------------------------------#
# API Routes for Document Upload, Question Answering, and Session Management

@router.post("/documents/upload")
def upload_document(
    source: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    return process_document_upload(
        source=source,
        file=file,
        db=db
    )



#----------------------------------------------------------------------------------------------------------#
# This Route is used to start a new session. It takes no input and returns the session_id.

@router.post("/sessions")
def start_session(
    db: Session = Depends(get_db)
):

    session = create_session(
        db
    )

    return {
        "session_id":
            session.session_id
    }

