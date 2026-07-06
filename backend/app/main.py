import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from .config import settings
from .search_engine import SearchEngine
from .chatbot import ECommerceChatbot
from .data_pipeline import DataPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intelligent E-commerce Sales Assistant API",
    description="A FastAPI backend supporting semantic search, product recommendation, comparison, and LLM-powered sales chat.",
    version="1.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to React origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Service Instances
search_engine = SearchEngine()
chatbot = ECommerceChatbot(search_engine)

@app.on_event("startup")
def startup_event():
    """Initializes search engine indices on application startup."""
    try:
        search_engine.initialize()
        logger.info("Search Engine successfully loaded and initialized during startup.")
    except Exception as e:
        logger.error(f"Failed to initialize search engine at startup: {str(e)}. Please initialize via /api/initialize.")

# Request/Response Schemas
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of sender: 'user' or 'assistant'")
    content: str = Field(..., description="Message contents")

class ChatRequest(BaseModel):
    query: str = Field(..., description="Current user query")
    history: List[ChatMessage] = Field(default=[], description="Chat history context")

class InitResponse(BaseModel):
    status: str
    message: str
    product_count: int

@app.post("/api/initialize", response_model=InitResponse)
def initialize_database(background_tasks: BackgroundTasks):
    """Triggers the synthetic data generation, pipeline cleaning, and FAISS indexing."""
    try:
        pipeline = DataPipeline()
        cleaned_df = pipeline.run()
        
        # Reload search engine with new data and build FAISS index
        search_engine.initialize(force_rebuild=True)
        
        # Re-initialize chatbot with updated search engine
        global chatbot
        chatbot = ECommerceChatbot(search_engine)
        
        return InitResponse(
            status="success",
            message="Database generated and FAISS index successfully rebuilt.",
            product_count=len(cleaned_df)
        )
    except Exception as e:
        logger.exception("Error initializing database:")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {str(e)}")

@app.post("/api/chat")
def chat_endpoint(request: ChatRequest):
    """Conversational endpoint returning chatbot response, products, comparison details, etc."""
    try:
        # Convert Pydantic schemas to standard dictionaries
        history_list = [{"role": msg.role, "content": msg.content} for msg in request.history]
        
        # Query chatbot conversational assistant
        chat_response = chatbot.chat(request.query, history_list)
        return chat_response
    except Exception as e:
        logger.exception("Error processing chat message:")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@app.get("/api/products")
def get_products(category: Optional[str] = None, top_k: int = 12):
    """Retrieves list of products, optionally filtered by category."""
    try:
        if search_engine.df is None:
            search_engine.initialize()
            
        df = search_engine.df
        if category:
            df = df[df["category"].str.lower() == category.lower()]
            
        products = df.head(top_k).to_dict(orient="records")
        # Clean NaN values
        for p in products:
            for key, val in p.items():
                if isinstance(val, float) and (val is None or val != val): # NaN check
                    p[key] = ""
                    
        return {"products": products}
    except Exception as e:
        logger.exception("Error listing products:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/categories")
def get_categories():
    """Returns list of unique product categories and stats in stock."""
    try:
        if search_engine.df is None:
            search_engine.initialize()
            
        counts = search_engine.df["category"].value_counts().to_dict()
        categories = [{"name": name, "count": int(count)} for name, count in counts.items()]
        return {"categories": categories}
    except Exception as e:
        logger.exception("Error listing categories:")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
@app.get("/api")
@app.get("/api/")
def health_check():
    """Health check endpoint showing backend status."""
    if search_engine.df is None or search_engine.df.empty:
        try:
            search_engine.initialize()
            global chatbot
            chatbot = ECommerceChatbot(search_engine)
        except Exception:
            pass
            
    is_db_ready = search_engine.df is not None and not search_engine.df.empty
    is_faiss_ready = search_engine.index is not None
    
    return {
        "status": "online",
        "database_ready": is_db_ready,
        "faiss_ready": is_faiss_ready,
        "product_count": len(search_engine.df) if is_db_ready else 0,
        "gemini_enabled": settings.GEMINI_API_KEY != ""
    }
