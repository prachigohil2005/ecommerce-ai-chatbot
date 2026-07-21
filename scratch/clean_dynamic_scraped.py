import pandas as pd
from pathlib import Path
import sys

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

from app.search_engine import SearchEngine

def clean():
    clean_csv = backend_dir / "data" / "cleaned_products.csv"
    raw_csv = backend_dir / "data" / "raw_products.csv"
    
    # 1. Clean cleaned_products.csv
    if clean_csv.exists():
        print("Cleaning cleaned_products.csv...")
        df = pd.read_csv(clean_csv)
        initial_len = len(df)
        
        # We can drop rows where subcategory is fashion dynamic
        df = df[df["subcategory"] != "fashion dynamic"]
        final_len = len(df)
        
        df.to_csv(clean_csv, index=False)
        print(f"Removed {initial_len - final_len} dynamic items from cleaned_products.csv.")
        
    # 2. Clean raw_products.csv
    if raw_csv.exists():
        print("Cleaning raw_products.csv...")
        raw_df = pd.read_csv(raw_csv)
        initial_len_raw = len(raw_df)
        raw_df = raw_df[raw_df["subcategory"] != "fashion dynamic"]
        final_len_raw = len(raw_df)
        
        raw_df.to_csv(raw_csv, index=False)
        print(f"Removed {initial_len_raw - final_len_raw} dynamic items from raw_products.csv.")
        
    # 3. Rebuild search index
    print("\nRebuilding FAISS index...")
    se = SearchEngine()
    se.initialize(force_rebuild=True)
    print("FAISS index successfully rebuilt!")

if __name__ == "__main__":
    clean()
