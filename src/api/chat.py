import time

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
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

    finally:

        latency = (time.perf_counter() - start) * 1000

        logger.info(
            "Chat Request Finished | Time=%.2f ms",
            latency,
        )




@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
):

    async def event_stream():

        start = time.perf_counter()

        logger.info(
            "Streaming Chat Started | Session=%s",
            request.session_id,
        )

        try:

            async for token in Agent.execute_stream(
                question=request.question,
                session_id=request.session_id,
                db=db,
            ):
                yield f"data: {token}\n\n"

            yield "data: [DONE]\n\n"

        except Exception:

            logger.exception(
                "Streaming Chat Failed | Session=%s",
                request.session_id,
            )

            yield "data: ERROR\n\n"

        finally:

            latency = (
                time.perf_counter() - start
            ) * 1000

            logger.info(
                "Streaming Chat Finished | Session=%s | Time=%.2f ms",
                request.session_id,
                latency,
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )