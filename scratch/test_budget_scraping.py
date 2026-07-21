import os
import sys
from pathlib import Path
import logging

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "backend"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=project_root / "backend" / ".env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from app.search_engine import SearchEngine
from app.chatbot import ECommerceChatbot

def test():
    se = SearchEngine()
    bot = ECommerceChatbot(se)
    
    query = "give me pink lehnga like anushka sharma in range of 25000 to 50000"
    print(f"\n--- Testing chat for budget query: '{query}' ---")
    
    # Enable logging for scraper and chatbot
    logging.getLogger("chatbot").setLevel(logging.INFO)
    logging.getLogger("scraper").setLevel(logging.INFO)
    logging.getLogger("search_engine").setLevel(logging.INFO)
    
    res = bot.chat(query, [])
    print("\n--- Response ---")
    print(res.get("response"))
    print("\n--- Products Returned ---")
    for idx, p in enumerate(res.get("products", [])):
         print(f"#{idx+1}: {p.get('product_title')} | Price: {p.get('selling_price')} | Score: {p.get('final_ranking_score')} | Alt: {p.get('is_alternative')} | Relaxed: {p.get('relaxed_filters')}")

if __name__ == "__main__":
    test()
