import os
import sys
import logging
from pathlib import Path

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_beachwear")

from app.chatbot import ECommerceChatbot
from app.search_engine import SearchEngine
from app.scraper import FlipkartScraper

def test_beachwear():
    logger.info("Initializing search engine, chatbot, and scraper...")
    engine = SearchEngine()
    engine.initialize()
    chatbot = ECommerceChatbot(engine)
    scraper = FlipkartScraper()
    
    user_query = "casual beach wear for women"
    logger.info(f"User Query: '{user_query}'")
    
    # Get NLU fallback extraction to see what search query it generates
    meta = chatbot.extract_entities_and_intent(user_query, [])
    search_query = meta.get("search_query")
    logger.info(f"Fallback NLU extracted search_query: '{search_query}'")
    
    # 1. Try scraping with the NLU expanded search_query (the concatenated string)
    logger.info(f"Scraping Flipkart using NLU expanded query: '{search_query}'...")
    expanded_results = scraper.scrape_query(query=search_query, category="Fashion", limit=3)
    logger.info(f"Expanded query returned {len(expanded_results)} products.")
    
    # 2. Try scraping with the original user_query
    logger.info(f"Scraping Flipkart using original user query: '{user_query}'...")
    original_results = scraper.scrape_query(query=user_query, category="Fashion", limit=3)
    logger.info(f"Original user query returned {len(original_results)} products.")
    for idx, p in enumerate(original_results):
        logger.info(f"  [{idx+1}] Title: {p.get('product_title')} | Price: {p.get('selling_price')}")
        
    assert len(original_results) > 0, "Original query should return scraped beachwear!"
    logger.info("Test validation complete!")

if __name__ == "__main__":
    test_beachwear()
