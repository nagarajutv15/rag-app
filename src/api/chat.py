import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.models.database import get_db
from src.models.chat_request import ChatRequest
from src.models.chat_response import ChatResponse
from src.agents.agent import Agent
from src.utils.logger import logger


router = APIRouter(
    prefix="/chat",
    tags=["Agentic AI"],
)


@router.post(
    "",
    response_model=ChatResponse,
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
):

    start = time.perf_counter()

    logger.info(
        "Chat Request Started | Session=%s",
        request.session_id,
    )

    try:

        result = await Agent.execute(
            question=request.question,
            session_id=request.session_id,
            db=db,
        )

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Chat Request Completed | Time=%.2f ms",
            latency,
        )

        return ChatResponse(
            session_id=result["session_id"],
            answer=result["answer"],
        )

    except Exception:

        logger.exception("Chat Request Failed")

        raise
