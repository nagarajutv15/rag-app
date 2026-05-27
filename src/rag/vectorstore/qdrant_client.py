import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance
)


load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "employee_rag")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", 1536))


QDRANT_CLIENT = QdrantClient(
    host = QDRANT_HOST,
    port = QDRANT_PORT
)

collections = QDRANT_CLIENT.get_collections()

existing_names = [collection.name for collection in collections.collections]

if QDRANT_COLLECTION not in existing_names:

    QDRANT_CLIENT.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )

    print( f"{QDRANT_COLLECTION} created")

else:

    print(f"{QDRANT_COLLECTION} already exists")
    