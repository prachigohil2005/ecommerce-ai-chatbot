import pandas as pd
from pathlib import Path
import json

backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
clean_csv = backend_dir / "data" / "cleaned_products.csv"
df = pd.read_csv(clean_csv)

print(f"Total products: {len(df)}")
beach_rows = df[df["product_title"].str.contains("beach", case=False, na=False) | 
                df["description"].str.contains("beach", case=False, na=False)]
print(f"Products containing 'beach': {len(beach_rows)}")
for idx, row in beach_rows.head(5).iterrows():
    print(f"  - ID: {row['product_id']} | Title: {row['product_title']}")
