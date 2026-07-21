import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

# Load env variables manually
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "backend" / ".env")

from app.search_engine import SearchEngine
from app.chatbot import ECommerceChatbot

def test():
    se = SearchEngine()
    se.initialize()
    bot = ECommerceChatbot(se)
    
    query = "I am thing to give her hand purse"
    print(f"\n--- Testing Query: '{query}' ---")
    
    # 1. Extract metadata
    meta = bot.extract_entities_and_intent(query, [])
    print("Extracted metadata:")
    import pprint
    pprint.pprint(meta)
    
    # 2. Search
    intent = meta.get("intent", "recommendation")
    category = meta.get("category")
    min_budget = meta.get("min_budget")
    max_budget = meta.get("max_budget")
    brand = meta.get("brand")
    features = meta.get("features", [])
    lifestyle = meta.get("lifestyle")
    color = meta.get("color")
    size = meta.get("size")
    gender = meta.get("gender")
    
    retrieved_products = se.search(
        query=query,
        category_filter=category,
        min_budget_filter=min_budget,
        max_budget_filter=max_budget,
        brand_filter=brand,
        top_k=8,
        color_filter=color,
        size_filter=size,
        gender_filter=gender
    )
    
    print("\nRetrieved Products:")
    for idx, p in enumerate(retrieved_products[:5]):
        print(f"#{idx+1}: {p.get('product_title')} | Brand: {p.get('brand')} | Category: {p.get('category')} | Price: {p.get('selling_price')} | Score: {p.get('semantic_score')}")
        
    chat_response = bot.chat(query, [])
    print("\nChat Response:")
    print(chat_response.get("response"))
    print("\nChat Response Products:")
    for idx, p in enumerate(chat_response.get("products", [])):
        print(f"#{idx+1}: {p.get('product_title')} | Brand: {p.get('brand')} | Price: {p.get('selling_price')}")

if __name__ == "__main__":
    test()
