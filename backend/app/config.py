import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables
load_dotenv(dotenv_path=BASE_DIR / ".env")

class Settings:
    # API configuration
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash") # Stable default
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    LLAMA_API_KEY: str = os.getenv("LLAMA_API_KEY", "")
    
    # Paths
    DATA_RAW_PATH: Path = BASE_DIR / "data" / "raw_products.csv"
    DATA_CLEAN_PATH: Path = BASE_DIR / "data" / "cleaned_products.csv"
    FAISS_INDEX_DIR: Path = BASE_DIR / "faiss_index"
    
    # Embedding config
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Hybrid Ranking Weights
    HYBRID_WEIGHT_SEMANTIC: float = 0.45
    HYBRID_WEIGHT_INTENT: float = 0.30
    HYBRID_WEIGHT_RATING: float = 0.10
    HYBRID_WEIGHT_POPULARITY: float = 0.10
    HYBRID_WEIGHT_AVAILABILITY: float = 0.05
    
    # Server port
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()
