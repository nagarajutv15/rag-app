from pydantic import BaseModel
from src.llm.llm_service import llm
from src.rag.retrieval.hybrid_search import hybrid_search



MIN_RERANK_SCORE = 3.0

#----------------------------------------------------------------------------------------------------------#
# This function rewrites the query for better retrieval based on the original question and specified rules.


def rewrite_query(
    query: str
) -> str:

    prompt = f"""
You are a retrieval optimization agent.

Original Question:

{query}

Rewrite the query for better retrieval.

Rules:

- Preserve meaning.
- Expand abbreviations.
- Include important keywords.
- Optimize for enterprise document search.
- Return only the rewritten query.
"""

    result = llm.invoke(prompt)

    return result.content.strip()



#----------------------------------------------------------------------------------------------------------#
# This function performs adaptive retrieval of documents based on the query and department ID, with a specified number of attempts for query rewriting and relevance grading.

def adaptive_retrieve(
    query: str,
    department_id: str,
    max_attempts: int = 2
):

    current_query = query

    for attempt in range(max_attempts):

        print(f"\nAdaptive Attempt {attempt + 1}")

        print(f"Query: {current_query}")

        retrieved_docs = hybrid_search(
            query=current_query,
            department_id=department_id,
            top_k=3
        )

        if not retrieved_docs:
            print("No documents found.")
            current_query = rewrite_query(current_query)
            continue

        top_score = retrieved_docs[0].get("rerank_score",0.0)

        print(f"Top Rerank Score: {top_score}")

        if top_score >= MIN_RERANK_SCORE:
            print("Retrieval accepted.")
            return retrieved_docs

        print("Low confidence retrieval.")

        current_query = rewrite_query(current_query)

        print(f"Rewritten Query: {current_query}")

    print("Adaptive retrieval failed.")

    return None