import os
from typing import List
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

EMBEDDING_BATCH_SIZE = int(os.getenv(
    "EMBEDDING_BATCH_SIZE",
    "100",
))


def get_embedding_model():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        chunk_size=EMBEDDING_BATCH_SIZE,
    )


def generate_embeddings(chunks: List[Document]):

    if not chunks:
        return []
    
    embeddings_model = get_embedding_model()

    texts = [chunk.page_content for chunk in chunks]

    vectors = embeddings_model.embed_documents(texts=texts)

    return vectors
    