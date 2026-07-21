import pandas as pd
from pathlib import Path
import json

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
clean_csv = backend_dir / "data" / "cleaned_products.csv"
df = pd.read_csv(clean_csv)

saree_rows = df[df["product_title"].str.contains("Wastera", case=False, na=False)]
row = saree_rows.iloc[0].to_dict()

from backend.app.search_engine import SearchEngine
engine = SearchEngine()

print("Product Row Keys and Values:")
for k, v in row.items():
    print(f"  {k}: {v} (type: {type(v)})")

prod_gender = engine.get_product_gender(row)
print(f"\nDetermined prod_gender: '{prod_gender}'")

gender_filter = "women"
is_match = not (gender_filter == "women" and prod_gender not in ["women", "unisex"])
print(f"Passes gender filter 'women': {is_match}")
