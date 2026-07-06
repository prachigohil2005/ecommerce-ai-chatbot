import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

# Load env variables manually
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "backend" / ".env")

import google.generativeai as genai
from app.config import settings
from app.search_engine import SearchEngine
from app.chatbot import ECommerceChatbot

def test():
    print("Gemini API Key:", os.getenv("GEMINI_API_KEY"))
    print("Gemini Model:", os.getenv("GEMINI_MODEL"))
    
    # Initialize SearchEngine (but don't need index for metadata test, wait, chatbot init calls search_engine.initialize() maybe? No, let's see)
    se = SearchEngine()
    bot = ECommerceChatbot(se)
    
    queries = [
        "gogles for women",
        "sunscreen for dry skin"
    ]
    
    for q in queries:
        print(f"\n--- Testing Query: '{q}' ---")
        meta = bot.extract_entities_and_intent(q, [])
        print("Extracted metadata:")
        import pprint
        pprint.pprint(meta)
        
        # Check is_vague logic
        intent = meta.get("intent", "recommendation")
        category = meta.get("category")
        budget = meta.get("budget")
        brand = meta.get("brand")
        features = meta.get("features", [])
        lifestyle = meta.get("lifestyle")
        
        is_vague = (
            (len(q.split()) < 3 and not category and not lifestyle) or 
            (not category and intent in ["recommendation", "filtering"]) or
            (category and not brand and not budget and not lifestyle and not features)
        )
        print(f"Is vague: {is_vague}")

if __name__ == "__main__":
    test()
