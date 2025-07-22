import time
import logging
import pandas as pd
import os

from src.config import Config
from src.api_clients import ElasticsearchClient, GenAIClient
from src.data_processing import add_embeddings_to_products, read_search_terms_from_file
from src.search_engine import SearchService

logger = logging.getLogger(__name__)

def main():
    """
    Main function to orchestrate the vector search testing process.
    """
    logger.info("Starting the semantic search testing process...")

    try:
        # Initialize Clients
        es_client = ElasticsearchClient(Config.ES_HOST, Config.ES_API_KEY)
        genai_client = GenAIClient(
            base_url=Config.GENAI_URL,
            api_key=Config.GENAI_API_KEY
        )
        search_service = SearchService(es_client, genai_client)

        # 1. Collect products from Elasticsearch
        products_to_process = es_client.get_products(Config.OLD_ES_INDEX_NAME, size=500)
        if not products_to_process:
            logger.warning("No products collected. Exiting.")
            return

        # 2. Generate embeddings for the products
        products_with_embeddings = add_embeddings_to_products(products_to_process, genai_client)
        if not products_with_embeddings:
            logger.warning("No embeddings generated. Exiting.")
            return

        # 3. Index products with embeddings
        es_client.index_documents(Config.NEW_ES_INDEX_NAME, products_with_embeddings)

        # 4. Read search terms from a file
        search_terms_file_path = os.path.join(os.path.dirname(__file__), '..', Config.SEARCH_TERMS_FILE)
        search_terms = read_search_terms_from_file(search_terms_file_path)
        if not search_terms:
            logger.warning("No search terms read. Exiting.")
            return

        # 5. Perform searches and log results
        all_results = []
        for term in search_terms:
            logger.info(f"\nPerforming searches for term: '{term}'")

            # Semantic Search
            start_time_sem = time.time()
            df_semantic = search_service.run_semantic_search(term, Config.NEW_ES_INDEX_NAME)
            end_time_sem = time.time()
            time_sem = end_time_sem - start_time_sem

            result_sem = {
                'searched_term': term,
                'search_type': 'Semantic',
                'execution_time_s': time_sem,
                'result_product_names': df_semantic['Product Name'].tolist() if not df_semantic.empty else []
            }
            all_results.append(result_sem)

            csv_filename_sem = f"results/{term.replace(' ', '_')}_semantic.csv"
            os.makedirs(os.path.dirname(csv_filename_sem), exist_ok=True)
            df_semantic.to_csv(csv_filename_sem, index=False)
            logger.info(f"  Semantic results for '{term}' saved to '{csv_filename_sem}'.")

            # Hybrid Search
            start_time_hyb = time.time()
            df_hybrid = search_service.run_hybrid_search(term, Config.NEW_ES_INDEX_NAME)
            end_time_hyb = time.time()
            time_hyb = end_time_hyb - start_time_hyb

            result_hyb = {
                'searched_term': term,
                'search_type': 'Hybrid',
                'execution_time_s': time_hyb,
                'result_product_names': df_hybrid['Product Name'].tolist() if not df_hybrid.empty else []
            }
            all_results.append(result_hyb)

            csv_filename_hyb = f"results/{term.replace(' ', '_')}_hybrid.csv"
            os.makedirs(os.path.dirname(csv_filename_hyb), exist_ok=True)
            df_hybrid.to_csv(csv_filename_hyb, index=False)
            logger.info(f"  Hybrid results for '{term}' saved to '{csv_filename_hyb}'.")

            # Lexical Search
            start_time_lex = time.time()
            df_lexical = search_service.run_lexical_search(term, Config.NEW_ES_INDEX_NAME)
            end_time_lex = time.time()
            time_lex = end_time_lex - start_time_lex

            result_lex = {
                'searched_term': term,
                'search_type': 'Lexical',
                'execution_time_s': time_lex,
                'result_product_names': df_lexical['Product Name'].tolist() if not df_lexical.empty else []
            }
            all_results.append(result_lex)

            csv_filename_lex = f"results/{term.replace(' ', '_')}_lexical.csv"
            os.makedirs(os.path.dirname(csv_filename_lex), exist_ok=True)
            df_lexical.to_csv(csv_filename_lex, index=False)
            logger.info(f"  Lexical results for '{term}' saved to '{csv_filename_lex}'.")

            logger.info("-" * 50)

        # Save a summary of all results
        df_all_results = pd.DataFrame(all_results)
        summary_csv_filename = "results/summary_vector_search_tests.csv"
        os.makedirs(os.path.dirname(summary_csv_filename), exist_ok=True)
        df_all_results.to_csv(summary_csv_filename, index=False)
        logger.info(f"\nOverall test summary saved to '{summary_csv_filename}'.")
        logger.info("Vector search testing process completed successfully!")

    except ValueError as ve:
        logger.critical(f"Configuration error: {ve}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during main execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()