import time
import uuid
import config
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

class RAGService:
    def __init__(self):
        print(f"📚 Loading Multilingual Model ({config.EMBEDDING_MODEL_NAME})...")
        self.encoder = SentenceTransformer(config.EMBEDDING_MODEL_NAME,token=False)
        
        print("🌲 Connecting to Pinecone...")
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        
        self.current_index_name = config.PINECONE_INDEX_NAME
        self.switch_index(self.current_index_name)

    def switch_index(self, index_name: str):
        """Connects to a specific index, creating it if it doesn't exist."""
        print(f"🔄 Switching to Pinecone Index: {index_name}...")
        try:
            existing_indexes = [i.name for i in self.pc.list_indexes()]
            if index_name not in existing_indexes:
                print(f"🌲 Creating new index: {index_name}...")
                self.pc.create_index(
                    name=index_name,
                    dimension=384, 
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                time.sleep(2)
            
            self.index = self.pc.Index(index_name)
            self.current_index_name = index_name
            return True, f"Connected to {index_name}"
        except Exception as e:
            return False, str(e)

    def list_indexes(self):
        try:
            return [i.name for i in self.pc.list_indexes()]
        except: return []

    def chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += (chunk_size - overlap)
        return chunks

    def ingest_file(self, filename: str, content: str):
        print(f"📄 Processing {filename}...")
        chunks = self.chunk_text(content)
        
        vectors = []
        for chunk in chunks:
            embedding = self.encoder.encode(chunk).tolist()
            metadata = {"filename": filename, "text": chunk}
            vector_id = str(uuid.uuid4())
            vectors.append({"id": vector_id, "values": embedding, "metadata": metadata})
            
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i+batch_size]
            self.index.upsert(vectors=batch)
            
        return len(chunks)

    def search(self, query: str, top_k=3):
        """
        Smart Search:
        1. If query is Non-English, translate it to English (better for English medical docs).
        2. Embed and Search.
        """
        search_query = query
        
        # Simple detection: If we detect non-ASCII characters, it might be Arabic/other
        # Or you can pass the 'language' param from the frontend if you updated main.py to pass it.
        # Here we use a robust auto-translation trick:
        
        try:
            # Check if query contains Arabic or high-unicode chars
            is_non_english = any(ord(char) > 127 for char in query) 
            
            if is_non_english:
                print(f"🌍 Detected non-English query: '{query}'")
                # Translate to English for better retrieval against English DB
                translated = GoogleTranslator(source='auto', target='en').translate(query)
                print(f"🔤 Translated for Search: '{translated}'")
                search_query = translated
        except Exception as e:
            print(f"⚠️ Translation warning: {e}")

        # Embed the (potentially translated) query
        query_embedding = self.encoder.encode(search_query).tolist()
        
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        contexts = [match['metadata']['text'] for match in results['matches']]
        return "\n\n".join(contexts)

rag_engine = RAGService()