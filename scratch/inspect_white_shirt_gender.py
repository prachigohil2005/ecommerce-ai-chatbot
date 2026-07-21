import pandas as pd
from pathlib import Path
import sys

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

clean_csv = backend_dir / "data" / "cleaned_products.csv"
df = pd.read_csv(clean_csv)

from app.search_engine import SearchEngine
se = SearchEngine()

shirt_rows = df[df["product_title"].str.contains("Vebnor", case=False, na=False)]

print(f"Found {len(shirt_rows)} Vernor products:")
for idx, row in shirt_rows.iterrows():
    p = row.to_dict()
    gender = se.get_product_gender(p)
    print(f"Title: {p['product_title']}")
    print(f"Category: {p['category']}")
    print(f"Gender determined: {gender}")
    print(f"Description: {p.get('description', '')[:200]}")
    print("-" * 50)
