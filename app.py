import os
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from src.helper import get_pinecone_vectorstore

load_dotenv()

app = FastAPI()

# --- 1. Link Static Files (CSS) ---
# This tells FastAPI: "If someone asks for /static/..., look in the static folder"
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- 2. Setup Templates (HTML) ---
templates = Jinja2Templates(directory="templates")

# --- 3. Initialize Database ---
PINECONE_API_KEY=os.getenv("pinecone")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
INDEX_NAME = "medical-chatbot"

# Load DB once at startup
docsearch = get_pinecone_vectorstore(PINECONE_API_KEY, INDEX_NAME)

# --- 4. Routes ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Serves the Homepage (HTML)
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/search")
async def search_endpoint(query: str = Form(...)):
    """
    Performs the Vector Search and returns raw results
    """
    print(f"🔎 Searching for: {query}")
    
    # Search Pinecone (Top 3 results)
    results = docsearch.similarity_search(query, k=3)
    
    # Format the results to send back to frontend
    search_data = [
        {"content": doc.page_content, "source": doc.metadata.get("source", "Unknown")}
        for doc in results
    ]
    
    return search_data