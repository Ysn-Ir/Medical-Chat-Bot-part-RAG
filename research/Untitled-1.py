# %%

pip install langchain

# %%
import os 
os.chdir('../')

# %%
import langchain
print(langchain.__version__)
import torch 
print(torch.__version__)


# %%
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyMuPDFLoader


# %%
def load_pdf_files(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf",
                           loader_cls=PyPDFLoader )
    documents =loader.load()
    return documents

# %%
extracted_data=load_pdf_files("data")

# %%
extracted_data[0:5]

# %%
len(extracted_data)

# %%
from typing import List
from langchain.schema import Document


# %%
def filter_text(docs: List[Document]) -> List[Document]:
    return [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source")}
        )
        for doc in docs
    ]


# %%
minimal_docs=filter_text(extracted_data)

# %%
minimal_docs[0:5]

# %%
def text_split(data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    texts_chunks=text_splitter.split_documents(data)
    return texts_chunks

# %%
text_chunks=text_split(minimal_docs)
print(len(text_chunks))

# %%
pip install --upgrade transformers

# %%
from langchain_community.embeddings import HuggingFaceEmbeddings

# %%
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")  # or "cuda"

# Embed a query
vector = model.encode("Hello world", normalize_embeddings=True)
print(vector[:10])


# %%
def download_emb():
    model_name = "BAAI/bge-small-en-v1.5"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings


# %%

import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available! GPU can be used.")
    print("GPU device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Using CPU.")


# %%
embeddings=download_emb()

# %%
embeddings

# %%
vec=embeddings.embed_query("sup yssdfSfSfefdsfassine")

# %%
len(vec)

# %%
from dotenv import load_dotenv
import os
load_dotenv()


# %%
PINECONE_API_KEY=os.getenv("pinecone")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

# %%
from pinecone import Pinecone 
pincone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pincone_api_key)

# %%
pc

# %%
from pinecone import ServerlessSpec
index_name = "medical-chatbot"
if not pc.has_index(index_name):
    pc.create_index(
        index_name,
        dimension=384,
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index=pc.Index(index_name)

# %%
from langchain_pinecone import PineconeVectorStore


docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

# %%
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,   # your embeddings object
    index_name=index_name   # name of your existing Pinecone index
)


# %%
query = "cold feet"

results = docsearch.similarity_search(
    query=query,
    k=5
)


# %%
for doc in results:
    print(doc.page_content)
    print(doc.metadata)



