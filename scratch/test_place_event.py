import logging
from backend.app.search_engine import SearchEngine
from backend.app.chatbot import ECommerceChatbot

# Set logging to warning to keep output clean
logging.basicConfig(level=logging.WARNING)

def test_queries():
    print("Initializing Search Engine...")
    engine = SearchEngine()
    engine.initialize()
    
    print("Initializing Chatbot...")
    bot = ECommerceChatbot(engine)
    
    queries = [
        "rajasthani western wear for women",
        "kashmir traditional winter wear for men",
        "garba traditional dress for women",
        "school wear and accessories for boys",
        "office interview clothes and accessories for men"
    ]
    
    for q in queries:
        print("\n" + "="*60)
        print(f"USER QUERY: {q}")
        print("="*60)
        res = bot.chat(q, [])
        
        print(f"EXTRACTED METADATA:")
        print(f"  Place: {res.get('place')}")
        print(f"  Weather Context: {res.get('weather_context')}")
        print(f"  Cultural Style: {res.get('cultural_style')}")
        print(f"  Occasion/Festival: {res.get('occasion_or_festival')}")
        print(f"  Event Requirements: {res.get('event_requirements')}")
        print(f"  Gender: {res.get('gender')}")
        
        print("\nBOT RESPONSE:")
        print(res.get('response'))
        print("-"*60)

if __name__ == "__main__":
    test_queries()
