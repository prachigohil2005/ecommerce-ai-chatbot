import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_school_bag")

from app.chatbot import ECommerceChatbot
from app.search_engine import SearchEngine

def test_school_bag():
    logger.info("Initializing search engine and chatbot...")
    engine = SearchEngine()
    engine.initialize()
    chatbot = ECommerceChatbot(engine)
    
    query = "tommor i am going to school"
    logger.info(f"Query under test: '{query}'")
    
    # Mock extract_entities_and_intent to return the correct LLM NLU mapping
    mock_nlu = {
        'intent': 'general query',
        'is_out_of_scope': False,
        'comparison_targets': [],
        'place': None,
        'weather_context': None,
        'cultural_style': None,
        'occasion_or_festival': 'school',
        'event_requirements': 'school bags',
        'exclude_brands': [],
        'negated_features': [],
        'color': None,
        'size': None,
        'gender': None,
        'clarification_questions': None,
        'category': 'Fashion',
        'min_budget': None,
        'max_budget': None,
        'search_query': 'school bag'
    }
    
    with patch.object(ECommerceChatbot, "extract_entities_and_intent", return_value=mock_nlu):
        # Execute query
        res = chatbot.chat(query, [])
        logger.info(f"Assistant Response:\n{res.get('response')}\n")
        
        logger.info("Recommended products in response:")
        for idx, p in enumerate(res.get("products", [])):
            logger.info(f"  - [{idx+1}] Title: {p.get('product_title')} | Price: ₹{p.get('selling_price')}")
            
        # Verify that we recommended school bags
        bag_found = False
        for p in res.get("products", []):
            title = str(p.get("product_title", "")).lower()
            if "bag" in title or "backpack" in title or "wildcraft" in title or "skybags" in title:
                bag_found = True
                
        assert bag_found, "BUG: Scraped school bags are not matching and recommended!"
        logger.info("Validation PASSED successfully!")

if __name__ == "__main__":
    test_school_bag()
