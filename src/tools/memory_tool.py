import time

from src.models.database import SessionLocal
from src.services.memory_service import get_chat_history
from src.utils.logger import logger


class MemoryTool:

    @staticmethod
    def execute(state):

        start = time.perf_counter()

        session_id = state.get("session_id")

        logger.info(
            "Memory Tool Started | Session=%s",
            session_id,
        )

        if not session_id:

            logger.info(
                "Memory Tool Completed | No Session"
            )

            return {
                "context": "",
                "sources": []
            }

        db = SessionLocal()

        try:

            history = get_chat_history(
                db=db,
                session_id=session_id,
                limit=5,
            )

            summary = history["summary"]

            messages = history["messages"]

            recent_conversation = "\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages
            )

            if summary:

                context = f"""
Conversation Summary
--------------------------------------------------

{summary}

--------------------------------------------------
Recent Conversation
--------------------------------------------------

{recent_conversation}
"""

            else:

                context = f"""
Recent Conversation
--------------------------------------------------

{recent_conversation}
"""

            latency = (time.perf_counter() - start) * 1000

            logger.info(
                "Memory Tool Completed | Messages=%d | Time=%.2f ms",
                len(messages),
                latency,
            )

            return {
                "context": context,
                "sources": []
            }

        except Exception:

            logger.exception(
                "Memory Tool Failed | Session=%s",
                session_id,
            )

            raise

        finally:

            db.close()