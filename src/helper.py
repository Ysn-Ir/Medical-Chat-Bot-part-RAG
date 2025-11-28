import os
import torch
from typing import List

# --- 1. LangChain & Data Processing Imports ---
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings # Updated import
from langchain_pinecone import PineconeVectorStore

# --- 2. Pinecone Client Import ---
from pinecone import Pinecone  # <--- CRITICAL: Fixes "NameError: Pinecone is not defined"

# --- 3. Model & Transformers Imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

import os
from pinecone import Pinecone
from langchain_text_splitters import RecursiveCharacterTextSplitter # <--- NEW LOCATION
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_core.documents import Document # <--- NEW LOCATION

# ==========================================
# PART A: DATA INGESTION FUNCTIONS
# (Only needed if you run ingestion code)
# ==========================================

def load_pdf_files(data_path):
    loader = DirectoryLoader(
        data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

def text_split(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(data)
    return text_chunks

def download_emb():
    """
    Downloads the embedding model.
    NOTE: Using 'all-MiniLM-L6-v2' to match your Pinecone Dimension (384).
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, # Use 'cuda' if you have GPU memory to spare for embeddings
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings

# ==========================================
# PART B: RAG & INFERENCE FUNCTIONS
# (Used by app.py for the Chatbot)
# ==========================================

def get_pinecone_vectorstore(api_key: str, index_name: str):
    
    os.environ["PINECONE_API_KEY"] = api_key
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return docsearch
def load_medical_model(base_model_path: str, adapter_path: str):
    """
    Loads the Tokenizer, Base Mistral Model, and LoRA Adapter.
    """
    print(f"⏳ Loading Tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"⏳ Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map="auto",
        local_files_only=True
    )

    print(f"⏳ Loading LoRA Adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float16,
        local_files_only=True
    )
    model.eval()
    
    return tokenizer, model

def build_rag_prompt(user_query: str, context_text: str) -> str:
    """
    Formats the prompt with the retrieved medical context.
    """
    return (
        "You are a medical AI assistant designed to give general wellness guidance only.\n"
        "Use the following Context Information to answer the patient's question.\n"
        "Do not provide diagnoses, do not name illnesses, and do not suggest medical tests.\n"
        f"### Context Information:\n{context_text}\n\n"
        "### Instructions:\n"
        "Speak in simple, calm language. Encourage rest and fluids.\n"
        f"### Patient: {user_query}\n"
        "### Assistant:"
    )

def generate_answer(model, tokenizer, prompt: str, max_tokens: int = 250) -> str:
    """
    Generates the text response from the loaded model.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    
    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up: Remove the original prompt from the answer
    if prompt in full_response:
        return full_response.replace(prompt, "").strip()
    return full_response