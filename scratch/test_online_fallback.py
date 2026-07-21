import sys
import os
import logging
import pandas as pd

# Configure logging to see output clearly
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.search_engine import SearchEngine
from backend.app.chatbot import ECommerceChatbot
from backend.app.config import settings

def test_online_fallback():
    logger.info("Initializing search engine...")
    engine = SearchEngine()
    engine.initialize()
    
    # Store initial record count
    initial_count = len(engine.df)
    logger.info(f"Initial product count in dataset: {initial_count}")
    
    bot = ECommerceChatbot(engine)
    
    # Test query that is guaranteed NOT to exist in the database (since men's clothes are only Navy Blue)
    test_query = "suggest brown shirt for men"
    logger.info(f"Sending test chat query: '{test_query}'")
    
    # Run chat process (this should trigger the Selenium scraper fallback)
    response = bot.chat(test_query, [])
    
    # Check if records were added
    final_count = len(engine.df)
    logger.info(f"Final product count in dataset: {final_count}")
    
    logger.info("Verifying chatbot response and retrieved products:")
    logger.info(f"Intent detected: {response.get('intent')}")
    logger.info(f"Products recommended: {len(response.get('products', []))}")
    
    # Verify we got brown shirts
    found_brown = False
    for p in response.get("products", []):
        logger.info(f"- Product: {p.get('product_title')} | Color: {p.get('color')} | Price: ₹{p.get('selling_price')}")
        if "brown" in str(p.get("color", "")).lower() or "brown" in str(p.get("product_title", "")).lower():
            found_brown = True
            
    if final_count > initial_count:
        logger.info(f"✓ Success: {final_count - initial_count} new products were dynamically scraped and added to the dataset!")
    else:
        logger.error("✗ Failure: No products were added to the dataset.")
        sys.exit(1)
        
    if found_brown:
        logger.info("✓ Success: Found products matching the requested 'Brown' color constraint.")
    else:
        logger.warning("! Warning: Scraped products did not explicitly contain 'Brown' in color specifications or title, but check titles above.")

    # Re-run search directly on the engine to verify persistent FAISS indexing
    logger.info("Re-running direct search on the engine for 'brown shirt'...")
    search_results = engine.search("brown shirt for men", color_filter="Brown", top_k=2)
    if search_results and not search_results[0].get("is_alternative"):
        logger.info(f"✓ Success: Direct FAISS search found the newly indexed product strictly: {search_results[0].get('product_title')}")
    else:
        logger.error("✗ Failure: Persistent FAISS search could not retrieve the new items strictly.")
        sys.exit(1)
        
    logger.info("✓ All online fallback tests passed successfully!")

if __name__ == "__main__":
    test_online_fallback()
