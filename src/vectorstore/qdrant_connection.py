import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)

from src.utils.logger import logger


load_dotenv()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QDRANT_HOST = os.getenv(
    "QDRANT_HOST",
    "localhost",
)

QDRANT_PORT = int(
    os.getenv(
        "QDRANT_PORT",
        6333,
    )
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "organization",
)

VECTOR_SIZE = int(
    os.getenv(
        "VECTOR_SIZE",
        1536,
    )
)

logger.info(
    "Connecting to Qdrant | Host=%s | Port=%d",
    QDRANT_HOST,
    QDRANT_PORT,
)

try:

    # ---------------------------------------------------------
    # Connect
    # ---------------------------------------------------------

    QDRANT_CLIENT = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False,
    )

    logger.info(
        "Connected to Qdrant"
    )

    # ---------------------------------------------------------
    # Collection Check
    # ---------------------------------------------------------

    collections = QDRANT_CLIENT.get_collections()

    existing_names = [
        collection.name
        for collection in collections.collections
    ]

    if QDRANT_COLLECTION not in existing_names:

        QDRANT_CLIENT.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )

        logger.info(
            "Qdrant Collection Created | Collection=%s",
            QDRANT_COLLECTION,
        )

    else:

        logger.info(
            "Qdrant Collection Exists | Collection=%s",
            QDRANT_COLLECTION,
        )

except Exception:

    logger.exception(
        "Failed to Initialize Qdrant"
    )

    raise