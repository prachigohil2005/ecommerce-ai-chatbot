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

def test():
    se = SearchEngine()
    se.initialize()
    
    query = "navratri outfit for women"
    print(f"\n--- Testing Query: '{query}' ---")
    
    # 1. Look at FAISS top 150 hits raw
    print("\n--- FAISS top 10 raw hits ---")
    query_embedding = se.model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = se.index.search(query_embedding, 150)
    for rank, (score, idx) in enumerate(zip(distances[0][:10], indices[0][:10])):
        if idx == -1:
            continue
        product_id = se.metadata["product_ids"][idx]
        product_row = se.df[se.df["product_id"] == product_id]
        if not product_row.empty:
            p = product_row.iloc[0].to_dict()
            gender = se.get_product_gender(p)
            print(f"#{rank+1}: {p['product_title']} | Cat: {p['category']} | Gender: {gender} | Score: {score}")

    # Let's count how many of the top 150 are in Fashion and have gender women/unisex
    fashion_women_count = 0
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        product_id = se.metadata["product_ids"][idx]
        product_row = se.df[se.df["product_id"] == product_id]
        if not product_row.empty:
            p = product_row.iloc[0].to_dict()
            gender = se.get_product_gender(p)
            if p.get("category", "").lower() == "fashion" and gender in ["women", "unisex"]:
                fashion_women_count += 1
    print(f"\nTotal Fashion & Women/Unisex products in top 150: {fashion_women_count}")

    # 2. Run the search method with strict filters
    print("\n--- Running se.search(...) ---")
    results = se.search(query=query, gender_filter="women")
    print(f"Returned {len(results)} results:")
    for idx, p in enumerate(results):
        print(f"#{idx+1}: {p.get('product_title')} | Gender: {se.get_product_gender(p)} | Alternative: {p.get('is_alternative')} | Relaxed: {p.get('relaxed_filters')}")

if __name__ == "__main__":
    test()
