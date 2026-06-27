from typing import List
from uuid import uuid4
from langchain_core.documents import Document
from qdrant_client.models import PointStruct
from src.vectorstore.qdrant_connection import (
    QDRANT_CLIENT,
    QDRANT_COLLECTION
)
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue
)



#----------------------------------------------------------------------------------------------------------#
# This function stores the document chunks and their corresponding embeddings in the Qdrant vector store.

def store_vectors(
    chunks: List[Document],
    vectors: List[list]
):
    
    if not chunks or not vectors:
        return

    points = []

    for chunk, vector in zip(chunks, vectors):
        payload = build_payload(chunk)

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

    print(f"Vectors inserted: {len(points)}")



#----------------------------------------------------------------------------------------------------------#
# This function builds the payload for each document chunk to be stored in the vector store.

def build_payload(
    chunk: Document
):

    metadata = chunk.metadata
    
    return {
        "chunk_id": metadata.get("chunk_id"),
        "document_id": metadata.get("document_id"),
        "file_name": metadata.get("file_name"),
        "version": metadata.get("version", 1),
        "is_active": metadata.get("is_active", True),
        "text": chunk.page_content
    }


#----------------------------------------------------------------------------------------------------------#
# This function deletes all vectors associated with a specific document ID from the Qdrant vector store.



def delete_document_vectors(
    document_id: int
):

    QDRANT_CLIENT.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id
                    )
                )
            ]
        )
    )

    print(f"Deleted vectors for document {document_id}")