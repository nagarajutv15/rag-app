from typing import List


def build_prompt(
    question: str,
    context_chunks: List[str]
) -> str:

    context = "\n\n".join(
        context_chunks
    )

    prompt = f"""
You are an intelligent assistant for document question answering.

Rules:
1. Answer ONLY from the provided context.
2. Do not make up information.
3. If the answer is not available, say:
   "I couldn't find information in the documents."
4. Keep answers clear and concise.

Context:
-------------------
{context}
-------------------

Question:
{question}

Answer:
"""

    return prompt.strip()