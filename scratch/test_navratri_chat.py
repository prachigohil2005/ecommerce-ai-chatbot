import os
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "backend"))

# Load env variables manually
from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / "backend" / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from app.search_engine import SearchEngine
from app.chatbot import ECommerceChatbot

def test():
    se = SearchEngine()
    se.initialize()
    bot = ECommerceChatbot(se)
    
    query = "navratri outfit for women"
    print(f"\n--- Testing chatbot.chat('{query}') ---")
    res = bot.chat(query, [])
    
    print("\n--- Final Recommended Products ---")
    for idx, p in enumerate(res.get("products", [])):
        print(f"#{idx+1}: {p.get('product_title')} | Subcat: {p.get('subcategory')} | Price: {p.get('selling_price')} | Score: {p.get('final_ranking_score')}")

if __name__ == "__main__":
    test()
