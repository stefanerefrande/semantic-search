import logging
from typing import Any, Dict, List

import pandas as pd
from elasticsearch import TransportError

from src.api_clients import ElasticsearchClient, GenAIClient
from src.config import Config

logger = logging.getLogger(__name__)


class SearchService:
    """
    Encapsulates the logic for semantic, hybrid, and lexical search in Elasticsearch.
    """

    def __init__(self, es_client: ElasticsearchClient, genai_client: GenAIClient):
        self.es_client = es_client
        self.genai_client = genai_client

    def _generate_query_embedding(self, term: str) -> List[float] | None:
        """Generates the embedding for a search term."""
        query_embeddings = self.genai_client.generate_embeddings(
            texts=[term],
            model=Config.EMBEDDING_MODEL,
            dimensions=Config.EMBEDDING_DIMENSIONS
        )
        if not query_embeddings[0]:
            logger.warning(f"Could not generate embedding for term '{term}'.")
            return None
        return query_embeddings[0]

    def run_semantic_search(self, term: str, index_name: str) -> pd.DataFrame:
        """Executes a pure vector (KNN) search in Elasticsearch."""
        logger.info(f"  Executing Semantic search for '{term}' on index '{index_name}'...")

        query_embedding = self._generate_query_embedding(term)
        if query_embedding is None:
            return pd.DataFrame()

        body = {
            "knn": {
                "field": Config.EMBEDDING_FIELD_NAME,
                "query_vector": query_embedding,
                "k": Config.SEARCH_RESULTS_LIMIT,
                "num_candidates": Config.SEARCH_RESULTS_LIMIT * 10
            },
            "_source": ["name", "id", "productUrl"]  # Including more relevant fields
        }

        try:
            hits = self.es_client.perform_search(index_name, body, Config.SEARCH_RESULTS_LIMIT)
            results = [{
                'Product ID': hit['_source'].get('id'),
                'Product Name': hit['_source'].get('name'),
                'Product URL': hit['_source'].get('productUrl', 'N/A'),
                'Score': hit['_score']
            } for hit in hits]
            df = pd.DataFrame(results)
            logger.info(f"    Semantic search returned {len(results)} results.")
            return df
        except Exception as e:
            logger.error(f"    Error in Semantic search for '{term}': {e}")
            return pd.DataFrame()

    def run_hybrid_search(self, term: str, index_name: str) -> pd.DataFrame:
        """Executes a hybrid search (multi_match + KNN) in Elasticsearch."""
        logger.info(f"  Executing Hybrid search for '{term}' on index '{index_name}'...")

        query_embedding = self._generate_query_embedding(term)

        # Add lexical search (multi_match)

        should_queries = [{
            "multi_match": {
                "query": term,
                "fields": [
                    "name^10.0",
                    "description^5.0"
                ],
                "boost": 2.5
            }
        }]

        # Add vector search (KNN) if embedding was generated
        if query_embedding is not None:
            should_queries.append({
                "knn": {
                    "field": Config.EMBEDDING_FIELD_NAME,
                    "query_vector": query_embedding,
                    "k": Config.SEARCH_RESULTS_LIMIT,
                    "num_candidates": Config.SEARCH_RESULTS_LIMIT * 10,
                    "boost": 2.0
                }
            })
        else:
            logger.warning("    Query embedding not available for Hybrid search. Executing lexical search only.")

        body = {
            "query": {
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1  # At least one of the clauses must match
                }
            },
            "_source": ["name", "id", "productUrl"]
        }

        try:
            hits = self.es_client.perform_search(index_name, body, Config.SEARCH_RESULTS_LIMIT)
            results = [{
                'Product ID': hit['_source'].get('id'),
                'Product Name': hit['_source'].get('name'),
                'Product URL': hit['_source'].get('productUrl', 'N/A'),
                'Score': hit['_score']
            } for hit in hits]
            df = pd.DataFrame(results)
            logger.info(f"    Hybrid search returned {len(results)} results.")
            return df
        except Exception as e:
            logger.error(f"    Error in Hybrid search for '{term}': {e}")
            return pd.DataFrame()

    def run_lexical_search(self, term: str, index_name: str) -> pd.DataFrame:
        """Executes a lexical search (multi_match) in Elasticsearch."""
        logger.info(f"  Executing Lexical search for '{term}' on index '{index_name}'...")

        body = {
            "query": {
                "multi_match": {
                    "query": term,
                    "fields": [
                        "name^10.0",
                        "description^5.0"
                    ],
                    "boost": 2.5
                }
            },
            "_source": ["name", "id", "productUrl"]
        }

        try:
            hits = self.es_client.perform_search(index_name, body, Config.SEARCH_RESULTS_LIMIT)
            results = [{
                'Product ID': hit['_source'].get('id'),
                'Product Name': hit['_source'].get('name'),
                'Product URL': hit['_source'].get('productUrl', 'N/A'),
                'Score': hit['_score']
            } for hit in hits]
            df = pd.DataFrame(results)
            logger.info(f"    Lexical search returned {len(results)} results.")
            return df
        except Exception as e:
            logger.error(f"    Error in Lexical search for '{term}': {e}")
            return pd.DataFrame()