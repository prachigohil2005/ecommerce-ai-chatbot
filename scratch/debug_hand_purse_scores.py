import os
import sys
from pathlib import Path
import numpy as np
import json

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

# Load env variables manually
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "backend" / ".env")

from app.search_engine import SearchEngine

def test():
    se = SearchEngine()
    se.initialize()
    
    query = "I am thing to give her hand purse"
    print(f"\n--- Debugging query: '{query}' ---")
    
    # Generate query embedding
    query_embedding = se.model.encode([query], normalize_embeddings=True).astype("float32")
    
    # Query FAISS index (retrieve top 10 matches)
    distances, indices = se.index.search(query_embedding, 15)
    
    print("\nTop 15 FAISS Raw Matches (No filtering applied):")
    for rank, (score, idx) in enumerate(zip(distances[0], indices[0])):
        if idx == -1:
            continue
        product_id = se.metadata["product_ids"][idx]
        product_row = se.df[se.df["product_id"] == product_id]
        if product_row.empty:
            continue
        product = product_row.iloc[0].to_dict()
        semantic_score = float((score + 1.0) / 2.0)
        prod_gender = se.get_product_gender(product)
        print(f"Rank {rank+1}:")
        print(f"  Title: {product.get('product_title')}")
        print(f"  Category: {product.get('category')} | Brand: {product.get('brand')} | Gender: {prod_gender}")
        print(f"  Raw Score: {score:.4f} | Semantic Score: {semantic_score:.4f}")
        
    print("\n--- Running Search with Filters ---")
    results = se.search(
        query=query,
        category_filter="Fashion",
        gender_filter="women"
    )
    print(f"\nSearch results with category='Fashion', gender='women' ({len(results)} items):")
    for idx, p in enumerate(results):
        print(f"#{idx+1}: {p.get('product_title')} | Brand: {p.get('brand')} | Category: {p.get('category')} | Score: {p.get('semantic_score')}")

if __name__ == "__main__":
    test()
