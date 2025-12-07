from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import pypdf
import io
import shutil
import tempfile

from schemas import UserRequest, BotResponse ,IndexSwitchRequest
# Import engines
from model_service import llm_engine 
from rag_service import rag_engine
from vision_service import vision_engine
from voice_service import voice_engine
from deep_translator import GoogleTranslator


#login and database
from fastapi.staticfiles import StaticFiles
from fastapi import Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import models, database, auth

models.Base.metadata.create_all(bind=database.engine)



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




#auth
# --- AUTH ENDPOINTS ---
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/register")
async def register(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    # Check if user exists
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Create User
    hashed_pw = auth.get_password_hash(form_data.password)
    new_user = models.User(username=form_data.username, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    return {"message": "User created successfully"}

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token = auth.create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}

# Main Chat (Explicit path)
@app.get("/index.html", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

#  Vision Page Route
@app.get("/vision.html", response_class=HTMLResponse)
async def vision_page(request: Request):
    return templates.TemplateResponse("vision.html", {"request": request})

#  Voice Page Route
@app.get("/voice.html", response_class=HTMLResponse)
async def voice_page(request: Request):
    return templates.TemplateResponse("voice.html", {"request": request})
# About Page Route
@app.get("/about.html", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})
# --- UPDATED CHAT ENDPOINT (Protected & Logged) ---
@app.post("/chat", response_model=BotResponse)
async def chat(
    request: UserRequest, 
    # 1. SECURITY: Check valid token
    current_user: models.User = Depends(auth.get_current_user), 
    # 2. DATABASE: Get session
    db: Session = Depends(database.get_db)
):
    print(f"📩 User: {current_user.username} | Query ({request.language}): {request.message}")
    
    # --- STEP A: Input Translation (Sandwich Layer 1) ---
    english_query = request.message
    if request.language != "en":
        try:
            english_query = GoogleTranslator(source='auto', target='en').translate(request.message)
            print(f"🇺🇸 EN Query: {english_query}")
        except Exception as e:
            print(f"⚠️ Translation warning: {e}")

    # --- STEP B: RAG Search ---
    # Search using the English query to match medical docs
    context = rag_engine.search(english_query)
    
    # --- STEP C: Generate AI Response ---
    # The LLM generates in English
    english_response = llm_engine.generate(
        user_message=english_query,
        context=context,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        system_instruction=request.system_instruction
    )
    
    # --- STEP D: Output Translation (Sandwich Layer 2) ---
    final_response = english_response
    if request.language != "en":
        try:
            print(f"🌍 Translating response to {request.language}...")
            final_response = GoogleTranslator(source='en', target=request.language).translate(english_response)
        except Exception as e:
            print(f"❌ Translation Error: {e}")

    # --- STEP E: Save History to Database ---
    # We save the original message (in user's lang) and the final response (in user's lang)
    try:
        history_entry = models.ChatHistory(
            user_id=current_user.id,
            user_message=request.message,
            bot_response=final_response
        )
        db.add(history_entry)
        db.commit()
        print("💾 Chat saved to DB")
    except Exception as e:
        print(f"⚠️ Failed to save history: {e}")
    
    return BotResponse(response=final_response)






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






@app.post("/analyze_xray")
async def analyze_xray(file: UploadFile = File(...)):
    print(f"🩻 Analyzing X-Ray: {file.filename}")
    
    try:
        # Read file directly into memory
        contents = await file.read()
        
        # Run Vision Service
        result = vision_engine.process_image(contents)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message"))
            
        return result
        
    except Exception as e:
        print(f"❌ X-Ray Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")