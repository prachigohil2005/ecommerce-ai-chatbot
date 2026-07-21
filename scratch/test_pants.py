import os
import sys
import logging
from pathlib import Path

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_pants")

from app.chatbot import ECommerceChatbot
from app.search_engine import SearchEngine

def test_pants():
    logger.info("Initializing search engine and chatbot...")
    engine = SearchEngine()
    engine.initialize()
    chatbot = ECommerceChatbot(engine)
    
    # We will simulate a conversation history where the user previously asked for white shirts
    history = [
        {"role": "user", "content": "suggest white shirts"},
        {"role": "assistant", "content": "Here are some white shirts..."}
    ]
    
    query = "give pant also"
    logger.info(f"Query under test: '{query}' with history: {history}")
    
    res = chatbot.chat(query, history)
    logger.info(f"Assistant Response:\n{res.get('response')}\n")
    
    logger.info("Recommended products in response:")
    for idx, p in enumerate(res.get("products", [])):
        logger.info(
            f"  - [{idx+1}] Title: {p.get('product_title')} | Color: {p.get('color')} | "
            f"Subcat: {p.get('subcategory')} | Price: ₹{p.get('selling_price')}"
        )
        
if __name__ == "__main__":
    test_pants()
