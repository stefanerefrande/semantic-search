import logging
from typing import Any, List, Dict

import requests
from elasticsearch import Elasticsearch, ConnectionError, TransportError

from src.config import Config

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """
    Manages connections and operations with Elasticsearch.
    """

    def __init__(self, host: str, api_key: str | None = None):
        self.host = host
        self.es = self._create_client(host, api_key)

    def _create_client(self, host: str, api_key: str | None) -> Elasticsearch:
        """Creates and returns an Elasticsearch client instance."""
        try:
            if api_key:
                client = Elasticsearch(host, api_key=api_key, timeout=30)
            else:
                client = Elasticsearch(host, timeout=30)

            # Test connection
            if not client.ping():
                logger.error(f"Could not connect to Elasticsearch at {host}. Check URL and credentials.")
                raise ConnectionError(f"Failed to connect to Elasticsearch at {host}")
            logger.info(f"Successfully connected to Elasticsearch at {host}")
            return client
        except ConnectionError as e:
            logger.error(f"Connection error initializing Elasticsearch at {host}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Elasticsearch at {host}: {e}")
            raise

    def get_products(self, index_name: str, size: int = 100) -> List[Dict[str, Any]]:
        """Collects a specific number of products from an existing Elasticsearch index."""
        logger.info(f"Collecting {size} products from index '{index_name}'...")
        try:
            response = self.es.search(
                index=index_name,
                body={"size": size, "query": {"match_all": {}}}
            )
            products = [hit['_source'] for hit in response['hits']['hits']]
            logger.info(f"Collected {len(products)} products.")
            return products
        except TransportError as e:
            logger.error(f"Transport error collecting products from Elasticsearch: {e.info}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error collecting products from Elasticsearch: {e}")
            return []

    def index_documents(self, index_name: str, documents: List[Dict[str, Any]]):
        """Indexes a list of documents in Elasticsearch."""
        logger.info(f"Populating index '{index_name}' with {len(documents)} documents...")
        indexed_count = 0

        # Caution: For large volumes, consider using bulk API for more efficient indexing
        for document in documents:
            try:
                doc_id = str(document.get('id')) if document.get('id') else None  # Use document ID if it exists
                self.es.index(index=index_name, id=doc_id, document=document)
                indexed_count += 1
            except TransportError as e:
                logger.error(f"  Transport error indexing document {document.get('id', '')}: {e.info}")
            except Exception as e:
                logger.error(f"  Unexpected error indexing document {document.get('id', '')}: {e}")

        logger.info(f"  Total of {indexed_count} documents indexed in '{index_name}'.")

    def perform_search(self, index_name: str, body: Dict[str, Any], size: int) -> List[Dict[str, Any]]:
        """Executes a generic search in Elasticsearch and returns the results."""
        try:
            response = self.es.search(
                index=index_name,
                body=body,
                size=size
            )
            return response['hits']['hits']
        except TransportError as e:
            logger.error(f"Transport error executing search in Elasticsearch: {e.info}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error executing search in Elasticsearch: {e}")
            raise


class GenAIClient:
    """
    Manages calls to the GenAI API for embedding generation.
    """

    def __init__(self, base_url: str, api_key: str | None = None):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def generate_embeddings(self, texts: List[str], model: str, dimensions: int) -> List[List[float]] | None:
        """Generates embeddings for a list of texts in a single API call."""
        if not texts:
            return []

        content = {
            "instances": {
                "texts": texts
            },
            "parameters": {
                "model": model,
                "dimensions": dimensions,
                "encoding_format": "float"
            }
        }

        try:
            response = requests.post(
                url=self.base_url,
                headers=self.headers,
                json=content,
                timeout=Config.EMBEDDING_BATCH_SIZE * 5  # Increase timeout for larger batches
            )
            response.raise_for_status()

            embeddings_data = response.json().get('embeddings', [])
            return [item['values'] for item in embeddings_data if 'values' in item]
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error generating embeddings: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
            return None