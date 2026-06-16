from pydantic import BaseModel

from src.llm.llm_service import llm

from src.rag.retrieval.hybrid_search import hybrid_search

#----------------------------------------------------------------------------------------------------------#
# This Pydantic model defines the structure for grading the relevance of retrieved documents.

class RetrievalGrade(BaseModel):
    relevant: bool


#----------------------------------------------------------------------------------------------------------#
# This function grades the relevance of retrieved documents based on the question and context provided.


def grade_retrieval(
    question: str,
    context: str
) -> bool:

    structured_llm = (
        llm.with_structured_output(RetrievalGrade)
    )

    prompt = f"""
You are a retrieval evaluator.

Question:

{question}

Retrieved Context:

{context}

Evaluate:

1. Is the context relevant to the question?
2. Is there enough information to answer?
3. Would answering from this context
   cause hallucination?

Return:

relevant = true

or

relevant = false
"""

    result = structured_llm.invoke(prompt)

    return result.relevant




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

        context = "\n\n".join(

            doc.get("text", "")

            for doc in retrieved_docs
        )

        relevant = (
            grade_retrieval(
                question=query,
                context=context
            )
        )

        print(f"Relevant: {relevant}")

        if relevant:
            print("Retrieval accepted.")
            return retrieved_docs

        current_query = rewrite_query(current_query)
        
        print(f"Rewritten Query: {current_query}")

    print("Adaptive retrieval failed.")

    return None