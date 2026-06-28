import time

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    Form,
)
from sqlalchemy.orm import Session

from src.models.database import get_db
from src.services.document_service import process_document_upload
from src.services.memory_service import create_session
from src.utils.logger import logger


router = APIRouter()


# ----------------------------------------------------------------------------------------------------------
# Document Upload
# ----------------------------------------------------------------------------------------------------------

@router.post("/documents/upload")
def upload_document(
    source: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):

    start = time.perf_counter()

    logger.info(
        "Document Upload Started | File=%s | Source=%s",
        file.filename,
        source,
    )

    try:

        result = process_document_upload(
            source=source,
            file=file,
            db=db,
        )

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Document Upload Completed | File=%s | Time=%.2f ms",
            file.filename,
            latency,
        )

        return result

    except Exception:

        logger.exception(
            "Document Upload Failed | File=%s",
            file.filename,
        )

        raise


# ----------------------------------------------------------------------------------------------------------
# Create Session
# ----------------------------------------------------------------------------------------------------------

@router.post("/sessions")
def start_session(
    db: Session = Depends(get_db),
):

    start = time.perf_counter()

    logger.info("Session Creation Started")

    try:

        session = create_session(db)

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Session Created | Session=%s | Time=%.2f ms",
            session.session_id,
            latency,
        )

        return {
            "session_id": session.session_id
        }

    except Exception:

        logger.exception(
            "Session Creation Failed"
        )

        raise

