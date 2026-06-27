from src.rag.hybrid_search import hybrid_search


class RAGTool:

    @staticmethod
    def execute(state):

        query = (
            state.get("rewritten_question")
            or state["question"]
        )

        documents = hybrid_search(
            query=query,
            top_k=5,
        )

        if not documents:
            return ""

        return "\n\n".join(
            doc["text"]
            for doc in documents
        )