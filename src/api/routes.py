from sqlalchemy.orm import Session
from src.models.database import get_db
from src.rag.adaptive.query_classifier import classify_query
from src.services.document_service import process_document_upload
from src.services.chat_service import ask_rag_question
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



#----------------------------------------------------------------------------------------------------------#
# This Route is used to ask a question to the RAG system. It takes the session_id and query as input and returns the answer.

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


#----------------------------------------------------------------------------------------------------------#
# This Route is used to classify the query and return the route to which the query should be sent. It takes the query as input and returns the route.

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

#----------------------------------------------------------------------------------------------------------#
