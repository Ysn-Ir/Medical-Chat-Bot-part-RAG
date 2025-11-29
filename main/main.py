from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import pypdf
import io

from schemas import UserRequest, BotResponse
# Import BOTH engines
from model_service import llm_engine 
from rag_service import rag_engine

app = FastAPI(title="Mistral Medical RAG")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=BotResponse)
async def chat(request: UserRequest):
    print(f"📩 Query: {request.message}")
    
    # 1. Search Pinecone for context
    print("🔎 Searching documents...")
    context = rag_engine.search(request.message)
    if context:
        print(f"💡 Found context: {context[:100]}...")
    
    # 2. Generate Answer
    response_text = llm_engine.generate(
        user_message=request.message,
        context=context,
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )
    return BotResponse(response=response_text)

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF or TXT files allowed.")
    
    print(f"📂 Receiving file: {file.filename}")
    content = ""
    
    try:
        # Read content
        file_bytes = await file.read()
        
        if file.filename.endswith(".pdf"):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        else:
            content = file_bytes.decode("utf-8")
            
        # Send to RAG engine
        num_chunks = rag_engine.ingest_file(file.filename, content)
        
        return {"filename": file.filename, "chunks_added": num_chunks, "status": "success"}
    
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))