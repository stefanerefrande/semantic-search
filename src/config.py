import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """
    Class to manage project configurations, read from environment variables.
    """

    # Elasticsearch Configuration
    ES_HOST: str = os.getenv("ES_HOST", "http://localhost:9200")
    ES_API_KEY: str | None = os.getenv("ES_API_KEY")

    OLD_ES_INDEX_NAME: str = os.getenv("OLD_ES_INDEX_NAME", "products_source")
    NEW_ES_INDEX_NAME: str = os.getenv("NEW_ES_INDEX_NAME", "products_with_embeddings")
    EMBEDDING_FIELD_NAME: str = os.getenv("EMBEDDING_FIELD_NAME", "productEmbedding")
    EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", "512"))

    # GenAI Configuration
    GENAI_URL: str = os.getenv("GENAI_URL", "http://localhost:8000/embeddings")
    GENAI_API_KEY: str | None = os.getenv("GENAI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-model-v1")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))

    # Search Configuration
    SEARCH_TERMS_FILE: str = os.getenv("SEARCH_TERMS_FILE", "terms_to_search.txt")
    SEARCH_RESULTS_LIMIT: int = int(os.getenv("SEARCH_RESULTS_LIMIT", "5"))

    @classmethod
    def validate_config(cls):
        """Validates if essential configurations are present."""
        required_vars = [
            "ES_HOST", "GENAI_URL",
            "OLD_ES_INDEX_NAME", "NEW_ES_INDEX_NAME",
            "EMBEDDING_FIELD_NAME", "EMBEDDING_DIMENSIONS",
            "EMBEDDING_MODEL", "EMBEDDING_BATCH_SIZE",
            "SEARCH_TERMS_FILE", "SEARCH_RESULTS_LIMIT"
        ]

        missing_vars = [var for var in required_vars if not getattr(cls, var)]

        if missing_vars:
            logger.error(
                f"Configuration Error: The following essential environment variables are missing or empty: {', '.join(missing_vars)}")
            raise ValueError("Essential configurations are missing.")


# Optional: Call validation at application startup
Config.validate_config()