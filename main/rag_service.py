import time
import uuid
import config
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

class RAGService:
    def __init__(self):
        print("📚 Loading Embedding Model (all-MiniLM-L6-v2)...")
        
        # 🚨 FIX: Added token=False to force anonymous download
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME, token=False)
        
        print("🌲 Connecting to Pinecone...")
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        # Check if index exists, create if not (Serverless spec for free tier)
        existing_indexes = [i.name for i in self.pc.list_indexes()]
        if config.PINECONE_INDEX_NAME not in existing_indexes:
            print(f"🌲 Creating new index: {config.PINECONE_INDEX_NAME}...")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=384, # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            time.sleep(2) # Wait for initialization
            
        self.index = self.pc.Index(config.PINECONE_INDEX_NAME)
        print("✅ RAG Service Ready!")

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Simple helper to split text into chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def ingest_file(self, filename: str, content: str):
        """Process text, embed it, and upload to Pinecone"""
        print(f"📄 Processing {filename}...")
        chunks = self.chunk_text(content)
        
        vectors = []
        for chunk in chunks:
            # Create embedding
            embedding = self.encoder.encode(chunk).tolist()
            # Create metadata
            metadata = {"filename": filename, "text": chunk}
            # Create vector ID
            vector_id = str(uuid.uuid4())
            
            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})
            
        # Upload in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
            
        return len(chunks)

    def search(self, query: str, top_k=3):
        """Search Pinecone for relevant context"""
        query_embedding = self.encoder.encode(query).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = [match['metadata']['text'] for match in results['matches']]
        return "\n\n".join(contexts)

# Singleton instance
rag_engine = RAGService()