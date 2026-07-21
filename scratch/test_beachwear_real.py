import os
import sys
import logging
from pathlib import Path

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_beachwear_real")

from app.chatbot import ECommerceChatbot
from app.search_engine import SearchEngine

def test_beachwear_real():
    logger.info("Initializing search engine and chatbot...")
    engine = SearchEngine()
    engine.initialize()
    chatbot = ECommerceChatbot(engine)
    
    query = "casual beach wear for women"
    logger.info(f"Query under test: '{query}'")
    
    # Run chat query
    res = chatbot.chat(query, [])
    logger.info(f"Assistant Response:\n{res.get('response')}\n")
    
    logger.info("Recommended products in response:")
    for idx, p in enumerate(res.get("products", [])):
        logger.info(f"  - [{idx+1}] Title: {p.get('product_title')} | Price: ₹{p.get('selling_price')}")
        
    # Check that at least one of the recommended products is indeed beachwear (scraped dynamically)
    beach_found = False
    for p in res.get("products", []):
        title = str(p.get("product_title", "")).lower()
        if "beach" in title or "maxi" in title or "tiered" in title:
            beach_found = True
            
    assert beach_found, "BUG: Scraped beachwear products are not in the top recommended products list!"
    logger.info("Validation PASSED successfully!")

if __name__ == "__main__":
    test_beachwear_real()
