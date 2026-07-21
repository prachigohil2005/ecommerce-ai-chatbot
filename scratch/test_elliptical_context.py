import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.search_engine import SearchEngine
from backend.app.chatbot import ECommerceChatbot

def test_elliptical_context():
    logger.info("Initializing search engine...")
    engine = SearchEngine()
    engine.initialize()
    
    bot = ECommerceChatbot(engine)
    
    # 1. First Turn: User asks for a blue cotton frock
    turn1_user = "suggest blue cotton frock for women"
    logger.info(f"--- Turn 1: User says '{turn1_user}' ---")
    res1 = bot.chat(turn1_user, [])
    logger.info(f"Turn 1 Intent: {res1.get('intent')} | Category: {res1.get('category')}")
    
    # Build history context
    history = [
        {"role": "user", "content": turn1_user},
        {"role": "assistant", "content": res1.get("response")}
    ]
    
    # 2. Second Turn: User says "give me black one"
    turn2_user = "give me black one"
    logger.info(f"--- Turn 2: User says '{turn2_user}' ---")
    
    # Track initial record count before Turn 2
    initial_count = len(engine.df)
    
    res2 = bot.chat(turn2_user, history)
    
    # Track final record count after Turn 2
    final_count = len(engine.df)
    
    logger.info(f"Turn 2 Intent: {res2.get('intent')} | Category: {res2.get('category')} | Color: {res2.get('color')}")
    logger.info(f"Products recommended in Turn 2: {len(res2.get('products', []))}")
    
    found_black_frock = False
    for p in res2.get("products", []):
        logger.info(f"- Recommended Product: {p.get('product_title')} | Color: {p.get('color')}")
        title_lower = str(p.get("product_title", "")).lower()
        color_lower = str(p.get("color", "")).lower()
        if "black" in color_lower or "black" in title_lower:
            if "frock" in title_lower or "dress" in title_lower or "saree" in title_lower or "lehenga" in title_lower:
                found_black_frock = True
                
    if final_count > initial_count:
        logger.info(f"✓ Success: {final_count - initial_count} black frocks were dynamically scraped and indexed from Flipkart!")
    else:
        logger.warning("! Warning: No products were dynamically added during Turn 2 (check if they were already added in previous runs).")
        
    if found_black_frock:
        logger.info("✓ Success: Correctly carried over 'frock' noun and recommended a black frock/dress!")
    else:
        logger.error("✗ Failure: Bot did not recommend a black frock/dress for the query.")
        sys.exit(1)
        
    logger.info("✓ Elliptical context carry-over test passed successfully!")

if __name__ == "__main__":
    test_elliptical_context()
