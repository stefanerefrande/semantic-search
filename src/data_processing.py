import logging
from typing import Any, List, Dict

from src.api_clients import GenAIClient
from src.config import Config

logger = logging.getLogger(__name__)


def add_embeddings_to_products(
        products: List[Dict[str, Any]],
        genai_client: GenAIClient
) -> List[Dict[str, Any]]:
    """
    Adds embeddings to each product, using batch calls to the embedding API.
    Combines 'name' and 'description' to generate the product embedding.
    """
    logger.info(f"Starting embedding generation for {len(products)} products in batches...")
    processed_products = []

    # Prepare the list of texts and maintain a mapping to the original products
    texts_to_embed = []
    # We use the product ID to map back, assuming 'id' is unique
    product_id_map: Dict[Any, Dict[str, Any]] = {product.get('id', idx): product for idx, product in
                                                 enumerate(products)}

    product_ids_ordered_for_batch: List[Any] = []

    for product in products:
        # Concatenate relevant fields for product embedding
        text_to_embed = f"{product.get('name', '')} {product.get('description', {})}"

        # Ensure the text is not empty to avoid unnecessary calls or errors
        if text_to_embed.strip():
            texts_to_embed.append(text_to_embed)
            product_ids_ordered_for_batch.append(product.get('id', None))  # Store original ID to map back
        else:
            logger.warning(
                f"Product with ID '{product.get('id', 'N/A')}' has no name or description. Skipping for embedding.")

    # Process texts in batches
    for i in range(0, len(texts_to_embed), Config.EMBEDDING_BATCH_SIZE):
        batch_texts = texts_to_embed[i: i + Config.EMBEDDING_BATCH_SIZE]
        batch_original_ids = product_ids_ordered_for_batch[i: i + Config.EMBEDDING_BATCH_SIZE]

        logger.info(
            f"  Processing batch of {len(batch_texts)} texts "
            f"({i + 1}-{min(i + Config.EMBEDDING_BATCH_SIZE, len(texts_to_embed))}/{len(texts_to_embed)})..."
        )

        batch_embeddings = genai_client.generate_embeddings(
            texts=batch_texts,
            model=Config.EMBEDDING_MODEL,
            dimensions=Config.EMBEDDING_DIMENSIONS
        )

        if batch_embeddings and len(batch_embeddings) == len(batch_texts):
            for j, original_product_id in enumerate(batch_original_ids):
                if original_product_id is not None and original_product_id in product_id_map:
                    product = product_id_map[original_product_id]
                    product[Config.EMBEDDING_FIELD_NAME] = batch_embeddings[j]
                    processed_products.append(product)
                else:
                    logger.warning(
                        f"  WARNING: Product ID {original_product_id} not found in product map. Embedding not added.")
        else:
            logger.warning(
                f"    WARNING: Could not get embeddings for batch starting at {i}. "
                f"Products in this batch might not have embeddings."
            )
            # If embeddings couldn't be obtained, add original products without the embedding field
            for j, original_product_id in enumerate(batch_original_ids):
                if original_product_id is not None and original_product_id in product_id_map:
                    processed_products.append(product_id_map[original_product_id])

    logger.info(f"Total of {len(processed_products)} products processed with embeddings.")
    return processed_products


def read_search_terms_from_file(file_path: str) -> List[str]:
    """Reads search terms from a text file."""
    logger.info(f"Reading search terms from '{file_path}'...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(terms)} terms.")
        return terms
    except FileNotFoundError:
        logger.error(f"Search terms file '{file_path}' not found. Please create the file.")
        return []
    except Exception as e:
        logger.error(f"Error reading terms file '{file_path}': {e}")
        return []