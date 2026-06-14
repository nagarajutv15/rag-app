from fastapi import (
    APIRouter,
    Depends
)

from sqlalchemy.orm import Session

from src.models.database import (
    get_db
)

from src.services.memory_service import (
    create_session
)

router = APIRouter()


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