import os
import re
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from .config import settings
from .scraper import FlipkartScraper

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        self.raw_path = settings.DATA_RAW_PATH
        self.clean_path = settings.DATA_CLEAN_PATH
        os.makedirs(os.path.dirname(self.raw_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.clean_path), exist_ok=True)

    def clean_text(self, text: Any) -> str:
        """Remove HTML tags, multiple whitespace, and clean strings."""
        if pd.isna(text) or text is None:
            return ""
        text = str(text)
        # Strip HTML
        text = re.sub(r"<[^>]*>", " ", text)
        # Strip multiple spaces
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and standardizes the dataset according to Flipkart specifications."""
        logger.info("Starting Flipkart dataset cleaning pipeline...")

        # 1. Remove duplicate products by product_id or product_url
        initial_len = len(df)
        if "product_id" in df.columns:
            df = df.drop_duplicates(subset=["product_id"]).copy()
        else:
            df = df.drop_duplicates(subset=["product_title"]).copy()
        logger.info(f"Removed {initial_len - len(df)} duplicates. Remaining rows: {len(df)}")

        # 2. Clean Text Columns
        df["product_title"] = df["product_title"].apply(self.clean_text)
        df["description"] = df["description"].apply(self.clean_text)
        df["brand"] = df["brand"].apply(self.clean_text).replace("", "Generic")
        
        # 3. Standardize Categories
        df["category"] = df["category"].str.strip().str.capitalize()
        
        # 4. Handle Pricing (clean and normalize numbers)
        df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce").fillna(0.0)
        df["original_price"] = pd.to_numeric(df["original_price"], errors="coerce").fillna(df["selling_price"])
        
        # If original price is smaller than selling price, correct it
        df["original_price"] = np.where(df["original_price"] < df["selling_price"], df["selling_price"], df["original_price"])
        
        # Calculate discount percentage if missing or 0
        df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce").fillna(0.0)
        df["discount_percentage"] = np.where(
            (df["discount_percentage"] == 0) & (df["original_price"] > df["selling_price"]),
            round(((df["original_price"] - df["selling_price"]) / df["original_price"]) * 100.0, 1),
            df["discount_percentage"]
        )

        # 5. Handle Ratings and Reviews
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)
        df["availability"] = df["availability"].fillna("In Stock").str.strip()

        # 6. Parse and Clean specifications JSON field
        df["specifications"] = df["specifications"].apply(self._clean_specifications_json)

        # 7. Generate embedding text
        df["embedding_text"] = df.apply(self._generate_embedding_text, axis=1)

        logger.info("Flipkart dataset cleaning complete.")
        return df

    def _clean_specifications_json(self, spec_val: Any) -> str:
        """Validates specifications column is a valid JSON string, fixing missing or empty values."""
        if pd.isna(spec_val) or not spec_val or str(spec_val).strip() == "":
            return "{}"
            
        spec_str = str(spec_val).strip()
        try:
            # Check if it is valid JSON
            json.loads(spec_str)
            return spec_str
        except Exception:
            # Fallback if string is formatted as python dict
            try:
                # Replace single quotes with double quotes
                cleaned = spec_str.replace("'", '"')
                json.loads(cleaned)
                return cleaned
            except Exception:
                return "{}"

    def _generate_embedding_text(self, row: pd.Series) -> str:
        """Constructs an engineered text block representing all searchable facets of the product."""
        title = row.get("product_title", "")
        desc = row.get("description", "")
        cat = row.get("category", "")
        subcat = row.get("subcategory", "")
        brand = row.get("brand", "")
        price = row.get("selling_price", 0.0)
        rating = row.get("rating", 4.0)
        
        # Deserialise specifications to append them as text
        specs_str = ""
        spec_json = row.get("specifications", "{}")
        try:
            specs = json.loads(spec_json)
            if isinstance(specs, dict):
                specs_str = ", ".join([f"{k}: {v}" for k, v in specs.items()])
        except Exception:
            pass

        text = (
            f"Product: {title}. Brand: {brand}. Category: {cat} ({subcat}). "
            f"Price: Rs. {price:.0f}. Rating: {rating:.1f}/5. "
            f"Specifications: {specs_str}. Description: {desc}."
        )
        return text

    def run(self):
        """Executes cleaning pipeline. Runs scraper if raw dataset is empty."""
        logger.info("Initializing Data Pipeline execution...")

        df = None
        # Load raw products CSV if it exists
        if os.path.exists(self.raw_path):
            try:
                df = pd.read_csv(self.raw_path)
                logger.info(f"Loaded {len(df)} raw products from {self.raw_path}")
            except Exception as e:
                logger.error(f"Error loading raw CSV: {str(e)}")

        # If CSV is empty or missing, run FlipkartScraper
        if df is None or len(df) < 100:
            logger.info("Raw Flipkart dataset missing or too small. Executing Scraper...")
            scraper = FlipkartScraper()
            df = scraper.run()

        # Clean raw dataset
        cleaned_df = self.clean_dataset(df)

        # Save to cleaned path
        cleaned_df.to_csv(self.clean_path, index=False)
        logger.info(f"Saved cleaned dataset ({len(cleaned_df)} products) to {self.clean_path}")
        return cleaned_df

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()
