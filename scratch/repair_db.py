import os
import sys
import pandas as pd
import json
from pathlib import Path

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

from app.data_pipeline import DataPipeline
from app.search_engine import SearchEngine

def repair_database():
    print("Starting database repair for misclassified women's apparel...")
    
    clean_csv = backend_dir / "data" / "cleaned_products.csv"
    raw_csv = backend_dir / "data" / "raw_products.csv"
    
    women_words = ["saree", "sari", "sadi", "dress", "lehenga", "kurti", "frock", "gown", "maxi"]
    
    pipeline = DataPipeline()
    
    for csv_path in [raw_csv, clean_csv]:
        if not csv_path.exists():
            continue
            
        print(f"Repairing {csv_path.name}...")
        df = pd.read_csv(csv_path)
        repaired_count = 0
        
        for idx, row in df.iterrows():
            if row.get("category") == "Fashion":
                title_lower = str(row.get("product_title", "")).lower()
                if any(w in title_lower for w in women_words):
                    # Inspect specifications JSON
                    spec_val = row.get("specifications", "{}")
                    try:
                        specs = json.loads(spec_val)
                        if isinstance(specs, dict) and specs.get("gender") == "Men":
                            # Fix it
                            specs["gender"] = "Women"
                            df.at[idx, "specifications"] = json.dumps(specs)
                            
                            # Regenerate description
                            brand = row.get("brand", "Generic")
                            title = row.get("product_title")
                            cat = row.get("category")
                            desc = (
                                f"Buy the latest {title} online at best prices. This authentic product from {brand} "
                                f"is ideal for your daily {cat.lower()} requirements. Features key aspects: "
                                f"{', '.join([f'{k}: {v}' for k, v in specs.items()])}."
                            )
                            df.at[idx, "description"] = desc
                            
                            # Regenerate embedding text
                            price = row.get("selling_price", 0.0)
                            rating = row.get("rating", 4.0)
                            subcat = row.get("subcategory", "fashion dynamic")
                            specs_str = ", ".join([f"{k}: {v}" for k, v in specs.items()])
                            emb_text = (
                                f"Product: {title}. Brand: {brand}. Category: {cat} ({subcat}). "
                                f"Price: Rs. {price:.0f}. Rating: {rating:.1f}/5. "
                                f"Specifications: {specs_str}. Description: {desc}."
                            )
                            df.at[idx, "embedding_text"] = emb_text
                            
                            repaired_count += 1
                    except Exception as e:
                        print(f"Error parsing specs for row {idx}: {str(e)}")
                        
        print(f"Repaired {repaired_count} rows in {csv_path.name}.")
        df.to_csv(csv_path, index=False)
        
    print("Rebuilding FAISS index to apply repairs...")
    engine = SearchEngine()
    engine.initialize(force_rebuild=True)
    print("Database repair and FAISS rebuild complete!")

if __name__ == "__main__":
    repair_database()
