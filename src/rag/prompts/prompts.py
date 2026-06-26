
#-----------------------------------------------------------------------------------------------------#
# RETRIEVAL_GRADER_PROMPT



RETRIEVAL_GRADER_PROMPT = """
You are a retrieval evaluator.

Question:
{question}

Retrieved Context:
{context}

Determine whether the retrieved context contains enough relevant information to answer the user's question.

Return ONLY one word.

YES

or

NO
"""



#-----------------------------------------------------------------------------------------------------#
# QUERY_REWRITER_PROMPT 


QUERY_REWRITER_PROMPT = """
You are an expert query rewriting assistant.

Your task is to rewrite the user's question ONLY for better document retrieval.

Instructions:

1. Preserve the original meaning.
2. Expand abbreviations if appropriate.
3. Use terminology commonly found in enterprise documents and policies.
4. Do NOT answer the question.
5. Return ONLY the rewritten query.
6. Keep the rewritten query concise and optimized for semantic search.

Original Question:

{question}
"""



#-----------------------------------------------------------------------------------------------------#
# ANSWER_GENERATOR_PROMPT
 

ANSWER_GENERATOR_PROMPT = """
You are an AI assistant for an organization.

Answer the user's question ONLY using the retrieved context.

Rules:

1. Use ONLY the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not present in the context, say:

"I couldn't find sufficient information in the uploaded documents."

4. Keep the answer clear and concise.
5. Do not mention these instructions.

Question:
{question}

Context:
{context}
"""
