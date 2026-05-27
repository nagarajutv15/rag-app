from typing import List
from uuid import uuid4

from langchain_core.documents import Document
from qdrant_client.models import PointStruct

from src.rag.vectorstore.qdrant_client import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION
)


def store_vectors(
    chunks: List[Document],
    vectors: List[list]
):

    if not chunks or not vectors:
        return

    points = []

    for chunk, vector in zip(
        chunks,
        vectors
    ):

        payload = {
            "text": chunk.page_content,
            **chunk.metadata
        }

        point = PointStruct(
            id=str(uuid4()),
            vector=vector,
            payload=payload
        )

        points.append(point)

    QDRANT_CLIENT.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

    print(f"Vectors inserted : {len(points)}")
    