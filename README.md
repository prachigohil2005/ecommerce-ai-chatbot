# Intelligent Product Recommendation Chatbot: A Virtual Sales Associate for E-commerce

An intelligent, end-to-end conversational AI shopping assistant that acts like a human sales associate for an Amazon-like retail experience. The chatbot transitions away from keyword matching, interpreting vague lifestyle requirements, budget boundaries, and brand filters from natural language to query a FAISS vector database containing 3000+ products. It then returns side-by-side product comparisons, structured specs, and personalized recommendations accompanied by strengths, weaknesses, and alternatives.

---

## 🚀 Key Features
1. **Intent Classification & NLU**: Classifies queries into `recommendation`, `comparison`, `filtering`, `general query`, or `clarification`.
2. **Entity Extraction**: Pulls category, budget constraints, brands, desired specs/features, and lifestyle intents (e.g. *Goa trip*, *gym beginner*, *coding university*).
3. **Clarification Engine**: Detects missing critical information (such as budget or use-case) and prompts the customer with relevant follow-up questions.
4. **FAISS Vector Retrieval**: Employs the `sentence-transformers/all-MiniLM-L6-v2` encoder to match user search requests against product catalogs.
5. **Multi-Factor Ranking Engine**: Ranks products using a customized weighted scoring formula:
   $$\text{Final Score} = 0.45 \times \text{Semantic Similarity} + 0.20 \times \text{Rating} + 0.20 \times \text{Popularity (Log Reviews)} + 0.15 \times \text{Price Relevance}$$
6. **Comparison Matrix Generator**: Automatically extracts and aligns specifications side-by-side in a comparative table containing prices, ratings, key specs, pros, and cons.
7. **Conversational Response Engine**: Connects to the Google Gemini API to generate helpful, context-rich human-like conversations (with a fully functional offline fallback mode).

---

## 🛠️ Technology Stack
* **Backend Framework**: Python 3.12, FastAPI, Uvicorn
* **Vector Store**: FAISS (Facebook AI Similarity Search)
* **Embedding Model**: Sentence-Transformers `all-MiniLM-L6-v2`
* **LLM Model**: Google Gemini API (`gemini-1.5-flash` or `gemini-2.5-flash`)
* **Frontend**: React 18, Vite, Tailwind CSS v3, Lucide Icons, Axios
* **Database**: CSV/Pandas Dataframe (Data stored in SQLite or direct CSV files)

---

## 📁 Folder Structure
```text
project_1_shopbot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI entrypoint & router definition
│   │   ├── config.py          # Environment settings loader
│   │   ├── scraper.py         # Amazon India BeautifulSoup web scraper
│   │   ├── data_pipeline.py   # Dataset cleaning, standardization, & synthetic fallback
│   │   ├── search_engine.py   # FAISS Index generator & custom ranker
│   │   ├── chatbot.py         # Entity NLU, Clarifications, & Gemini interface
│   │   └── test_integration.py# Integration test verification suite
│   ├── data/
│   │   ├── raw_products.csv       # Unfiltered CSV dataset
│   │   └── cleaned_products.csv   # Normalized product CSV with embedding texts
│   ├── faiss_index/
│   │   ├── index.faiss        # Compiled FAISS Index database
│   │   └── index_metadata.pkl # Pickle metadata linking index to products
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Configuration file (Gemini API key, port)
├── frontend/
│   ├── src/
│   │   ├── App.jsx            # Main dashboard component (Chat, Explorer, Comparer)
│   │   ├── index.css          # Tailwind style imports & custom glassmorphism components
│   │   ├── main.jsx           # App mounting point
│   │   └── App.css            # Blank CSS reset
│   ├── tailwind.config.js     # Tailwind CSS configuration
│   ├── postcss.config.js      # PostCSS configuration
│   ├── vite.config.js         # Vite configuration with API Proxy setup
│   └── package.json           # Frontend Node modules configuration
└── README.md                  # System setup documentation
```

---

## ⚙️ Quick Start Installation & Execution

### 1. Set Up the Backend
First, navigate to the project directory and activate the virtual environment:
```bash
# Navigate to the workspace (already in project root)
cd /home/petpooja-1255/Downloads/project_1_shopbot

# Install backend dependencies (if running manually in a separate shell)
/home/petpooja-1255/venv/bin/pip install -r backend/requirements.txt
```

Create/modify `backend/.env` to include your Google Gemini API key:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
PORT=8000
```

Start the FastAPI development server:
```bash
# Run server from the root of the project
PYTHONPATH=. /home/petpooja-1255/venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API is now running on `http://localhost:8000`. You can inspect the interactive docs at `http://localhost:8000/docs`.

---

### 2. Set Up the Frontend (React + Vite)
Because Node.js is hosted locally, prepend its path to run commands:
```bash
# Set path to node bin
export PATH=/home/petpooja-1255/Downloads/node-v20.12.2-linux-x64/bin:$PATH

# Navigate to frontend folder
cd frontend

# Install package dependencies
npm install

# Run the frontend server locally
npm run dev
```
The React frontend application is now active on `http://localhost:5173`. Open it in your web browser.

---

## 🧪 Running Integration Tests
To verify that the FAISS index loading, vector search, price decay functions, and chatbot responders are fully functional, run the automated integration test:
```bash
PYTHONPATH=. /home/petpooja-1255/venv/bin/python -m backend.app.test_integration
```

---

## 🧠 Core System Design Details

### A. Intent Classification & Entity Extraction NLU
When a message arrives, the chatbot uses a strict system instruction context to query the LLM. It forces a JSON response structure that isolates the user's intent. If the LLM call fails or the user is offline, rule-based regular expression fallback algorithms extract the target budget constraints (e.g. `₹40,000` or `50k`) and match product category keywords (e.g. *phone*, *laptop*, *serum*, *shoes*).

### B. Custom Multi-Factor Ranking Engine
Traditional vector retrieval only matches description similarity. To mirror a real human sales associate, our ranking engine balances four parameters:
1. **Semantic Match (0.45 weight)**: Dot product cosine similarity of normalized embeddings.
2. **Review Rating (0.20 weight)**: User ratings normalized: $\frac{\text{Rating}}{5.0}$.
3. **Popularity (0.20 weight)**: Popularity is log-normalized against the database maximum review count: $\frac{\log(1+\text{reviews})}{\log(1+\text{max\_reviews})}$. This prevents hyper-popular items from completely overriding newer items, while still favoring trusted models.
4. **Price Relevance (0.15 weight)**: If a budget $B$ is specified, items under $B$ receive $1.0$. Items exceeding budget decay exponentially: $e^{-5.0 \times \frac{\text{Price} - B}{B}}$, ensuring slightly higher-priced items can be discussed as premium suggestions, while excessively expensive items are excluded.
