import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
)

from src.utils.logger import logger


load_dotenv()


# --------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------

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
    "Initializing Qdrant | Host=%s | Port=%d | Collection=%s",
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
)


# --------------------------------------------------------------------------------------------------
# Initialize Qdrant
# --------------------------------------------------------------------------------------------------

try:

    # ------------------------------------------------------------------
    # Connect
    # ------------------------------------------------------------------

    QDRANT_CLIENT = QdrantClient(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        check_compatibility=False,
    )

    logger.info(
        "Connected to Qdrant Successfully"
    )

    # ------------------------------------------------------------------
    # Verify Connection
    # ------------------------------------------------------------------

    collections = QDRANT_CLIENT.get_collections()

    logger.info(
        "Qdrant Connection Verified"
    )

    # ------------------------------------------------------------------
    # Check Collection
    # ------------------------------------------------------------------

    existing_collections = {

        collection.name

        for collection in collections.collections

    }

    if QDRANT_COLLECTION not in existing_collections:

        logger.info(
            "Collection Not Found. Creating | Collection=%s",
            QDRANT_COLLECTION,
        )

        QDRANT_CLIENT.create_collection(

            collection_name=QDRANT_COLLECTION,

            vectors_config=VectorParams(

                size=VECTOR_SIZE,

                distance=Distance.COSINE,

            ),

        )

        logger.info(
            "Collection Created Successfully | Collection=%s",
            QDRANT_COLLECTION,
        )

    else:

        logger.info(
            "Collection Already Exists | Collection=%s",
            QDRANT_COLLECTION,
        )

except Exception as ex:

    logger.exception(
        "Failed to Initialize Qdrant | Host=%s | Port=%d | Collection=%s",
        QDRANT_HOST,
        QDRANT_PORT,
        QDRANT_COLLECTION,
    )

    raise RuntimeError(
        "Unable to initialize Qdrant."
    ) from ex

finally:

    logger.info(
        "Qdrant Initialization Finished"
    )