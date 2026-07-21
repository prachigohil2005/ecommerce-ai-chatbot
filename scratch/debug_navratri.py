import os
import sys
from pathlib import Path
import json
import pprint

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "backend"))

# Load env variables manually
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / "backend" / ".env")

from app.search_engine import SearchEngine
from app.chatbot import ECommerceChatbot

def test():
    print("Initializing SearchEngine...")
    se = SearchEngine()
    se.initialize()
    print("Initializing Chatbot...")
    bot = ECommerceChatbot(se)
    
    query = "navratri outfit for women"
    print(f"\n--- Testing Query: '{query}' ---")
    meta = bot.extract_entities_and_intent(query, [])
    print("Extracted metadata:")
    pprint.pprint(meta)
    
    gender = meta.get("gender")
    print(f"Extracted Gender: {gender}")
    
    print("\n--- Running Search Engine ---")
    search_query = meta.get("search_query") or query
    print(f"Search query used: '{search_query}'")
    
    results = se.search(
        query=search_query,
        category_filter=meta.get("category"),
        min_budget_filter=meta.get("min_budget"),
        max_budget_filter=meta.get("max_budget"),
        brand_filter=meta.get("brand"),
        top_k=8,
        exclude_brands=meta.get("exclude_brands"),
        negated_features=meta.get("negated_features"),
        color_filter=meta.get("color"),
        size_filter=meta.get("size"),
        gender_filter=gender
    )
    
    print(f"Found {len(results)} results:")
    for idx, p in enumerate(results):
        print(f"#{idx+1}: {p.get('product_title')} | Gender determined: {se.get_product_gender(p)} | Color: {p.get('color')} | Subcat: {p.get('subcategory')} | Price: {p.get('selling_price')} | Score: {p.get('final_ranking_score')}")

if __name__ == "__main__":
    test()
