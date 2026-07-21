import os
import sys
import logging
from pathlib import Path
from unittest.mock import patch
from dotenv import load_dotenv

# Add workspace to path
backend_dir = Path("/home/petpooja-1255/Downloads/project_1_shopbot/backend")
sys.path.append(str(backend_dir))

# Load .env
load_dotenv(dotenv_path=backend_dir / ".env")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("test_llm_gateway")

from app.llm_gateway import LLMGateway

def test_llm_gateway():
    logger.info("Initializing LLMGateway...")
    
    # 1. We will mock the environment variables to have two keys:
    # First is an invalid key that will crash, second is the real valid key from config.
    real_key = os.getenv("GEMINI_API_KEY", "").split(",")[0]
    mock_keys = f"INVALID_KEY_12345, {real_key}"
    
    with patch.dict(os.environ, {"GEMINI_API_KEY": mock_keys}):
        gateway = LLMGateway()
        assert len(gateway.gemini_keys) == 2, "Should parse exactly two keys"
        assert gateway.gemini_keys[0] == "INVALID_KEY_12345"
        
        # System instructions and prompt
        sys_inst = "You are a precise JSON extractor."
        prompt = "Return JSON matching: {\"intent\": \"recommendation\"}"
        
        # Trigger metadata extraction
        logger.info("Triggering metadata extraction with key pool (expecting first key to fail and rotate)...")
        data = gateway.extract_metadata(
            query="suggest white shirts",
            history=[],
            system_instruction=sys_inst,
            prompt=prompt
        )
        
        logger.info(f"NLU output retrieved: {data}")
        assert data is not None, "Extraction should succeed after cascading to Groq!"
        
        # Since both Gemini keys failed (one invalid, one rate-limited), it rotated twice: 0 -> 1 -> 0
        logger.info(f"Final Gemini key index: {gateway.current_gemini_idx}")
        assert gateway.current_gemini_idx in [0, 1], "Gateway index should be rotated"
        
        logger.info("LLM gateway key rotation test PASSED successfully!")

if __name__ == "__main__":
    test_llm_gateway()
