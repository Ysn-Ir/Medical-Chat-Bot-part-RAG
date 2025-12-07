from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import pypdf
import io
import shutil
import tempfile
from voice_service import voice_engine
from schemas import UserRequest, BotResponse ,IndexSwitchRequest
# Import BOTH engines
from model_service import llm_engine 
from rag_service import rag_engine
from fastapi.staticfiles import StaticFiles
from deep_translator import GoogleTranslator

app = FastAPI(title="Mistral Medical RAG")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
def save_temp(upload_file):
    try:
        suffix = os.path.splitext(upload_file.filename)[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
            return tmp.name
    except: return None

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/indexes")
async def get_indexes():
    indexes = rag_engine.list_indexes()
    return {
        "indexes": indexes, 
        "current_index": rag_engine.current_index_name
    }

# 2. POST Endpoint to create a new index
@app.post("/create_index")
async def create_index(request: IndexSwitchRequest):
    print(f"🆕 Request to create index: {request.index_name}")
    # We reuse the switch_index logic because it handles creation automatically
    success, message = rag_engine.switch_index(request.index_name)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
        
    return {"status": "success", "message": f"Index '{request.index_name}' created and selected."}
@app.post("/set_index")
async def set_index(request: IndexSwitchRequest):
    print(f"⚙️ Request to switch index to: {request.index_name}")
    success, message = rag_engine.switch_index(request.index_name)
    
    if not success:
        raise HTTPException(status_code=500, detail=message)
        
    return {"status": "success", "message": message, "current_index": request.index_name}

@app.post("/chat", response_model=BotResponse)
async def chat(request: UserRequest):
    print(f"📩 Query ({request.language}): {request.message}")
    
    # 1. Translate User Query to English (for the LLM to understand)
    english_query = request.message
    if request.language != "en":
        try:
            english_query = GoogleTranslator(source='auto', target='en').translate(request.message)
            print(f"🇺🇸 Internal English Query: {english_query}")
        except:
            pass

    # 2. Search Pinecone (Using English Query) -> Returns English Context
    context = rag_engine.search(english_query)
    
    # 3. Generate Answer (All in English)
    english_response = llm_engine.generate(
        user_message=english_query,
        context=context,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system_instruction=request.system_instruction
    )
    print(f"🤖 AI Response (EN): {english_response[:50]}...")

    # 4. Final Step: Translate Answer back to User Language
    final_response = english_response
    if request.language != "en":
        try:
            print(f"🌍 Translating response to {request.language}...")
            final_response = GoogleTranslator(source='en', target=request.language).translate(english_response)
        except Exception as e:
            print(f"❌ Translation Error: {e}")
            # Fallback: return English if translation fails
    
    return BotResponse(response=final_response)

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
@app.post("/analyze_voice")
async def analyze_voice(
    cough: UploadFile = File(...),
    breath: UploadFile = File(...),
    speech: UploadFile = File(...)
):
    print("🎙️ Receiving voice samples...")
    
    # Helper to save upload to temp file
    def save_temp(upload_file):
        try:
            suffix = os.path.splitext(upload_file.filename)[1]
            if not suffix: suffix = ".wav" # Default to wav if missing
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(upload_file.file, tmp)
                return tmp.name
        except Exception as e:
            print(f"Error saving temp file: {e}")
            return None

    # Save all 3 files
    path_cough = save_temp(cough)
    path_breath = save_temp(breath)
    path_speech = save_temp(speech)

    if not all([path_cough, path_breath, path_speech]):
        raise HTTPException(status_code=500, detail="Failed to process audio files")

    try:
        # Run Analysis
        result = voice_engine.analyze_audio(path_cough, path_breath, path_speech)
        return result
        
    finally:
        # Cleanup: Delete temp files
        for p in [path_cough, path_breath, path_speech]:
            if p and os.path.exists(p):
                os.unlink(p)
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    temp_path = save_temp(file)
    if not temp_path:
        raise HTTPException(status_code=500, detail="Failed to save audio")
        
    try:
        text = voice_engine.transcribe(temp_path)
        return {"text": text}
    finally:
        if os.path.exists(temp_path): os.unlink(temp_path)

# --- ENDPOINT 2: Medical Diagnosis ---
@app.post("/diagnose")
async def diagnose_audio(
    cough: UploadFile = File(...),
    breath: UploadFile = File(...),
    speech: UploadFile = File(...)
):
    p_cough = save_temp(cough)
    p_breath = save_temp(breath)
    p_speech = save_temp(speech)

    if not all([p_cough, p_breath, p_speech]):
        raise HTTPException(status_code=500, detail="Failed to save audio files")

    try:
        result = voice_engine.diagnose(p_cough, p_breath, p_speech)
        return result
    finally:
        # Cleanup
        for p in [p_cough, p_breath, p_speech]:
            if p and os.path.exists(p): os.unlink(p)