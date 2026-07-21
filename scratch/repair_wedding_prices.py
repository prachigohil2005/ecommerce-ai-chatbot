import pandas as pd
from pathlib import Path
import sys

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

from app.search_engine import SearchEngine

def repair():
    clean_csv = backend_dir / "data" / "cleaned_products.csv"
    raw_csv = backend_dir / "data" / "raw_products.csv"
    
    # 1. Repair cleaned_products.csv
    if clean_csv.exists():
        df = pd.read_csv(clean_csv)
        
        # Repair Sherwani
        sherwani_mask = df["product_title"].str.contains("Sherwani", case=False, na=False) & (df["selling_price"] < 1000)
        df.loc[sherwani_mask, "selling_price"] = 2499.0
        df.loc[sherwani_mask, "original_price"] = 4999.0
        
        # Repair Wedding Suit
        suit_mask = df["product_title"].str.contains("Wedding Suit", case=False, na=False) & (df["selling_price"] < 1000)
        df.loc[suit_mask, "selling_price"] = 3499.0
        df.loc[suit_mask, "original_price"] = 6999.0
        
        df.to_csv(clean_csv, index=False)
        print(f"Repaired {sherwani_mask.sum()} Sherwanis and {suit_mask.sum()} Wedding Suits in cleaned_products.csv.")
        
    # 2. Repair raw_products.csv
    if raw_csv.exists():
        raw_df = pd.read_csv(raw_csv)
        
        sherwani_mask_raw = raw_df["product_title"].str.contains("Sherwani", case=False, na=False) & (raw_df["selling_price"] < 1000)
        raw_df.loc[sherwani_mask_raw, "selling_price"] = 2499.0
        raw_df.loc[sherwani_mask_raw, "original_price"] = 4999.0
        
        suit_mask_raw = raw_df["product_title"].str.contains("Wedding Suit", case=False, na=False) & (raw_df["selling_price"] < 1000)
        raw_df.loc[suit_mask_raw, "selling_price"] = 3499.0
        raw_df.loc[suit_mask_raw, "original_price"] = 6999.0
        
        raw_df.to_csv(raw_csv, index=False)
        print(f"Repaired {sherwani_mask_raw.sum()} Sherwanis and {suit_mask_raw.sum()} Wedding Suits in raw_products.csv.")
        
    # 3. Rebuild search index
    print("\nRebuilding FAISS index...")
    se = SearchEngine()
    se.initialize(force_rebuild=True)
    print("FAISS index successfully rebuilt!")

if __name__ == "__main__":
    repair()
