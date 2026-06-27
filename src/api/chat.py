from fastapi import APIRouter, Depends

from sqlalchemy.orm import Session

from src.models.database import get_db
from src.models.chat_request import ChatRequest
from src.models.chat_response import ChatResponse

from src.agents.agent import Agent


router = APIRouter(
    prefix="/chat",
    tags=["Agentic AI"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):

    result = Agent.execute(
        question=request.question,
        session_id=request.session_id,
        db=db,
    )

    return ChatResponse(
        session_id=result["session_id"],
        answer=result["answer"],
    )