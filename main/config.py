
import torch
from dotenv import load_dotenv
import os
load_dotenv()
# -----------------------------
# Configuration Settings
# -----------------------------
BASE_MODEL_PATH = r"C:\Users\khali\OneDrive\Bureau\mistral fine tuned\Mistral-7B-Instruct-v0.2"
LORA_REPO_ID = r"C:\Users\khali\OneDrive\Bureau\mistral fine tuned\mistral-medical-adapter-final"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

PINECONE_INDEX_NAME = "general-medstral"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"