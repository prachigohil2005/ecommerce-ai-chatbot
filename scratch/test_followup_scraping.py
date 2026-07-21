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
    
    # Turn 1
    q1 = "i am planing manali trip for traking"
    print(f"\n--- Turn 1: '{q1}' ---")
    res1 = bot.chat(q1, [])
    print("\nProducts Returned Turn 1:")
    for idx, p in enumerate(res1.get("products", [])):
         print(f"#{idx+1}: {p.get('product_title')} | Subcat: {p.get('subcategory')} | Price: {p.get('selling_price')}")
         
    # Turn 2
    history = [
        {"role": "user", "content": q1},
        {"role": "assistant", "content": res1.get("response")}
    ]
    q2 = "suggest clothes also"
    print(f"\n--- Turn 2: '{q2}' ---")
    res2 = bot.chat(q2, history)
    print("\nProducts Returned Turn 2:")
    for idx, p in enumerate(res2.get("products", [])):
         print(f"#{idx+1}: {p.get('product_title')} | Subcat: {p.get('subcategory')} | Price: {p.get('selling_price')}")

if __name__ == "__main__":
    test()
