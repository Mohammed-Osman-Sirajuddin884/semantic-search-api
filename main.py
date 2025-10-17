# main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Search API",
    description="An API that computes semantic similarity between documents and a query using embeddings.",
    version="1.0.0"
)

# Enable CORS (allowing all origins and headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

# Load a lightweight local embedding model
# This downloads the model only once on first run
print("🔄 Loading embedding model... (this may take ~10 seconds initially)")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Model loaded successfully!")

# Define request body schema
class RequestBody(BaseModel):
    docs: list[str]
    query: str

# Define /similarity endpoint
@app.post("/similarity")
def similarity(request: RequestBody):
    try:
        # Ensure we have documents
        if not request.docs or not request.query.strip():
            raise HTTPException(status_code=400, detail="Docs and query cannot be empty.")

        # Compute embeddings
        doc_embeddings = model.encode(request.docs)
        query_embedding = model.encode([request.query])[0]

        # Compute cosine similarities
        similarities = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Rank documents by similarity
        top_indices = np.argsort(similarities)[::-1][:3]
        top_docs = [request.docs[i] for i in top_indices]

        return {"matches": top_docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Optional root route
@app.get("/")
def root():
    return {"message": "Semantic Search API is running! Visit /docs for the interactive UI."}

