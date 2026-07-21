import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app.search_engine import SearchEngine
from backend.app.chatbot import ECommerceChatbot

def test_saree():
    logger.info("Initializing search engine...")
    engine = SearchEngine()
    engine.initialize()
    
    bot = ECommerceChatbot(engine)
    
    query = "red south indian saree"
    logger.info(f"Sending query: '{query}'")
    
    res = bot.chat(query, [])
    
    logger.info(f"Intent: {res.get('intent')}")
    logger.info(f"Category: {res.get('category')}")
    logger.info(f"Recommended products: {len(res.get('products', []))}")
    for idx, p in enumerate(res.get("products", [])):
        logger.info(f"[{idx+1}] Title: {p.get('product_title')}")
        logger.info(f"    Color: {p.get('color')} | Brand: {p.get('brand')} | Price: {p.get('selling_price')}")
        logger.info(f"    Semantic: {p.get('semantic_score'):.4f} | Rating: {p.get('rating_score'):.4f} | Popularity: {p.get('popularity_score'):.4f} | PriceRel: {p.get('price_relevance'):.4f} | Final: {p.get('final_ranking_score'):.4f}")
        
    logger.info("Directly searching search engine with color and category filters...")
    cat_results = engine.search(query, color_filter="Red", category_filter="Fashion", top_k=10)
    for idx, p in enumerate(cat_results):
        logger.info(f"Cat search [{idx+1}]: {p.get('product_title')} | Subcat: {p.get('subcategory')}")
        
    logger.info("Directly searching search engine with color, category, AND gender filters...")
    gender_results = engine.search(query, color_filter="Red", category_filter="Fashion", gender_filter="women", top_k=10)
    for idx, p in enumerate(gender_results):
        logger.info(f"Gender search [{idx+1}]: {p.get('product_title')} | Subcat: {p.get('subcategory')}")

if __name__ == "__main__":
    test_saree()
