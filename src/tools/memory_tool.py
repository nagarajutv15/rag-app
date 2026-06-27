from src.services.memory_service import get_chat_history
from src.models.database import SessionLocal


class MemoryTool:

    @staticmethod
    def execute(state):

        session_id = state.get("session_id")

        if not session_id:
            return ""

        db = SessionLocal()

        try:

            messages = get_chat_history(
                db=db,
                session_id=session_id,
                limit=10,
            )

            return "\n".join(
                f"{message['role']}: {message['content']}"
                for message in messages
            )

        finally:

            db.close()