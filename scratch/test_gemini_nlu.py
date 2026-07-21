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
import google.generativeai as genai

def test():
    api_key = os.getenv("GEMINI_API_KEY", "").split(",")[0].strip()
    print("Using API Key:", api_key[:10] + "...")
    genai.configure(api_key=api_key)
    
    se = SearchEngine()
    bot = ECommerceChatbot(se)
    
    query = "navratri outfit for women"
    print(f"\nExtracting metadata for query: '{query}'")
    
    # We will invoke extract_entities_and_intent
    try:
        meta = bot.extract_entities_and_intent(query, [])
        import pprint
        pprint.pprint(meta)
    except Exception as e:
        print("Error during extraction:", str(e))

if __name__ == "__main__":
    test()
