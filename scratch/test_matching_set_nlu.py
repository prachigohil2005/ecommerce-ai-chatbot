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
    
    query = "bride and groom mwching wedding clothes set"
    print(f"\nExtracting metadata for query: '{query}'")
    
    meta = bot.extract_entities_and_intent(query, [])
    import pprint
    pprint.pprint(meta)

if __name__ == "__main__":
    test()
