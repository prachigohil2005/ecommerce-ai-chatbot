import sys
import logging
import json
from backend.app.search_engine import SearchEngine
from backend.app.chatbot import ECommerceChatbot

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_integration_tests():
    logger.info("Starting Flipkart integration tests...")
    
    # 1. Initialize Search Engine
    try:
        engine = SearchEngine()
        engine.initialize()
        logger.info("✓ Search Engine initialized successfully.")
    except Exception as e:
        logger.error(f"✗ Search Engine initialization failed: {str(e)}")
        sys.exit(1)
        
    # 2. Test Semantic Search and Multi-factor Ranking
    try:
        query = "best gaming laptop under 100000"
        results = engine.search(query, budget_filter=100000.0, top_k=3)
        
        if not results:
            logger.error("✗ Semantic search returned 0 results.")
            sys.exit(1)
            
        logger.info(f"✓ Search test passed. Found {len(results)} matches for '{query}':")
        for i, res in enumerate(results):
            logger.info(f"  #{i+1}: {res['product_title']} - Price: Rs. {res['selling_price']} (Score: {res['final_ranking_score']:.4f})")
    except Exception as e:
        logger.error(f"✗ Search test failed: {str(e)}")
        sys.exit(1)
        
    # 3. Test Chatbot NLU & Responder
    try:
        bot = ECommerceChatbot(engine)
        chat_res = bot.chat("Suggest a high quality phone for gaming under 50000", [])
        
        logger.info("✓ Chatbot query processed successfully.")
        logger.info(f"  Detected Intent: {chat_res['intent']}")
        logger.info(f"  Detected Category: {chat_res['category']}")
        logger.info(f"  Detected Budget: {chat_res['budget']}")
        logger.info(f"  Number of Recommended Products: {len(chat_res['products'])}")
        logger.info("  Conversational Snippet:")
        snippet = "\n".join(chat_res['response'].split("\n")[:4])
        logger.info(f"\n{snippet}...")
        
    except Exception as e:
        logger.error(f"✗ Chatbot integration test failed: {str(e)}")
        sys.exit(1)
        
    logger.info("✓ All Flipkart integration tests passed successfully!")

if __name__ == "__main__":
    run_integration_tests()
