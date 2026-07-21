import sys
import os
sys.path.append('.')
from backend.app.search_engine import SearchEngine
from backend.app.chatbot import ECommerceChatbot

engine = SearchEngine()
# only need to test extract_query_intent
intent = engine.extract_query_intent("14 year old girl")
print("14 year old girl ->", intent.to_dict())

intent2 = engine.extract_query_intent("14 year old sis")
print("14 year old sis ->", intent2.to_dict())

intent3 = engine.extract_query_intent("teenage girl")
print("teenage girl ->", intent3.to_dict())
