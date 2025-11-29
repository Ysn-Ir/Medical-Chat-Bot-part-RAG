
from dotenv import load_dotenv
import os
from pinecone import Pinecone 
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from helper import load_load_pdf_files,filter_text,text_split,download_emb
load_dotenv()

PINECONE_API_KEY=os.getenv("pinecone")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

pincone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pincone_api_key)


index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        index_name,
        dimension=384,
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index=pc.Index(index_name)

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,   # your embeddings object
    index_name=index_name   # name of your existing Pinecone index
)
