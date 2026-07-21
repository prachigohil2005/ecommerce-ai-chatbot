import os
import sys
import logging
from pathlib import Path

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_price_range")

from app.chatbot import ECommerceChatbot
from app.search_engine import SearchEngine

def test_price_range():
    logger.info("Initializing search engine and chatbot...")
    engine = SearchEngine()
    engine.initialize()
    chatbot = ECommerceChatbot(engine)
    
    query = "white t shirt in range 2000 to 5000"
    logger.info(f"Query under test: '{query}'")
    
    # 1. Test parsing of budget range bounds via the regex fallback parser
    min_b, max_b = chatbot.parse_budget_range(query)
    logger.info(f"Rule-based extracted range bounds: min_budget={min_b}, max_budget={max_b}")
    assert min_b == 2000.0, f"Expected min_budget to be 2000.0, got {min_b}"
    assert max_b == 5000.0, f"Expected max_budget to be 5000.0, got {max_b}"
    logger.info("Rule-based range bound parsing validation PASSED!")

    # 2. Test chatbot metadata merging
    meta = chatbot.extract_entities_and_intent(query, [])
    logger.info(f"Full NLU parsed metadata: {meta}")
    assert meta.get("min_budget") == 2000.0, f"Expected NLU min_budget to be 2000.0, got {meta.get('min_budget')}"
    assert meta.get("max_budget") == 5000.0, f"Expected NLU max_budget to be 5000.0, got {meta.get('max_budget')}"
    logger.info("Full NLU range bound metadata extraction PASSED!")

    # 3. Search database directly with range constraints
    logger.info("Searching database using budget range constraints...")
    results = engine.search(
        query=meta.get("search_query") or "white t-shirt",
        category_filter="Fashion",
        min_budget_filter=2000.0,
        max_budget_filter=5000.0,
        color_filter="White",
        top_k=5
    )
    
    logger.info(f"Search retrieved {len(results)} products:")
    for idx, p in enumerate(results):
        logger.info(
            f"[{idx+1}] {p.get('product_title')} | Price: ₹{p.get('selling_price')} | "
            f"PriceRel: {p.get('price_relevance'):.4f} | Final Score: {p.get('final_ranking_score'):.4f}"
        )
    # Verify that all products in range (2000 to 5000) are ranked before products out of range
    in_range_items = [p for p in results if 2000.0 <= p.get("selling_price") <= 5000.0]
    out_of_range_items = [p for p in results if not (2000.0 <= p.get("selling_price") <= 5000.0)]
    
    logger.info(f"Verified {len(in_range_items)} items in range and {len(out_of_range_items)} items out of range.")
    assert len(in_range_items) > 0, "No in-range items were returned!"
    
    for ir in in_range_items:
        for oor in out_of_range_items:
            assert ir["final_ranking_score"] > oor["final_ranking_score"], \
                f"Item {ir['product_title']} (score: {ir['final_ranking_score']}) should rank above {oor['product_title']} (score: {oor['final_ranking_score']})"
    logger.info("Ranking order validation PASSED!")
    
    # 4. Chat response validation
    logger.info("Sending message to chatbot chat API...")
    res = chatbot.chat(query, [])
    logger.info(f"Assistant Response:\n{res.get('response')}\n")
    logger.info("Recommended products in response:")
    for idx, p in enumerate(res.get("products", [])):
        logger.info(f"  - Title: {p.get('product_title')} | Price: ₹{p.get('selling_price')}")
        
    logger.info("All verification checks PASSED successfully!")

if __name__ == "__main__":
    test_price_range()
