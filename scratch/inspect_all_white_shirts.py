import pandas as pd
from pathlib import Path
import sys

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

clean_csv = backend_dir / "data" / "cleaned_products.csv"
df = pd.read_csv(clean_csv)

from app.search_engine import SearchEngine
se = SearchEngine()

# Search for all products with "White Shirt" or "Vebnor" in title
matching_rows = df[df["product_title"].str.contains("White Shirt|Vebnor", case=False, na=False)]
print(f"Found {len(matching_rows)} matching products:")
for idx, row in matching_rows.iterrows():
    p = row.to_dict()
    gender = se.get_product_gender(p)
    print(f"ID: {p['product_id']} | Title: {p['product_title']} | Cat: {p['category']} | Gender determined: {gender}")
