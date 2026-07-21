import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "backend"))

# Load env variables manually
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / "backend" / ".env")

from app.search_engine import SearchEngine

def test():
    se = SearchEngine()
    se.initialize()
    
    # 1. Print total products
    print(f"Total products in DB: {len(se.df)}")
    
    # 2. Search for any products with 'purse' or 'bag' in title
    purse_df = se.df[se.df['product_title'].str.contains('purse|bag|clutch|handbag', case=False, na=False)]
    print(f"\nProducts with purse/bag/clutch/handbag in title ({len(purse_df)} items):")
    for idx, row in purse_df.head(10).iterrows():
        print(f"- {row['product_title']} | Category: {row['category']} | Gender: {row.get('gender')}")
        
    # 3. Let's see the last 5 added products in the dataframe
    print("\nLast 5 products added to the dataframe:")
    for idx, row in se.df.tail(5).iterrows():
        print(f"- {row['product_title']} | Category: {row['category']} | Gender: {row.get('gender')}")

if __name__ == "__main__":
    test()
