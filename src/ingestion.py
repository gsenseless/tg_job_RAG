import pdfplumber
from google.cloud import aiplatform
from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from vertexai.language_models import TextEmbeddingModel
import numpy as np
from typing import List, Dict
import pandas as pd
import time


class IngestionSubsystem:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005") #text-multilingual-embedding-002
        self.db = firestore.Client(project=project_id, database='ragdb')
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF resume."""
        text_parts = []
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    
    def get_embedding(self, text: str, max_retries: int = 5) -> List[float]:
        """Generate embedding using Vertex AI with retry logic."""
        for attempt in range(max_retries):
            try:
                embeddings = self.embedding_model.get_embeddings([text])
                return embeddings[0].values
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8, 16, 32 seconds
                    print(f"Rate limit hit. Waiting {wait_time} seconds... Error: {e}")
                    time.sleep(wait_time)
                else:
                    raise
        raise Exception("Max retries exceeded for embedding generation.")
    
    def get_embeddings_batch(self, texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """Generate embeddings for multiple texts in a single API call (up to 250)."""
        for attempt in range(max_retries):
            try:
                embeddings = self.embedding_model.get_embeddings(texts)
                return [emb.values for emb in embeddings]
            except Exception as e:
                if "429" in str(e) or "Quota exceeded" in str(e):
                    wait_time = (2 ** attempt) * 2
                    print(f"Rate limit hit. Waiting {wait_time} seconds... Error: {e}")
                    time.sleep(wait_time)
                else:
                    raise
        raise Exception("Max retries exceeded for batch embedding generation")
    
    def ingest_resume(self, pdf_file, user_id: str = "default_user") -> Dict:
        """Process resume: extract text and generate embedding."""
        resume_text = self.extract_text_from_pdf(pdf_file)
        resume_embedding = self.get_embedding(resume_text)
        
        doc_ref = self.db.collection("resumes").document(user_id)
        doc_ref.set({
            "text": resume_text,
            "embedding": resume_embedding,
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
        return {
            "user_id": user_id,
            "text_length": len(resume_text),
            "embedding_dim": len(resume_embedding)
        }
    
    def ingest_jobs_batch(self, jobs_data: List[Dict], session_id: str, progress_callback=None) -> List[Dict]:
        """Process jobs in batches of up to 100 (Firestore limit is 500, but smaller is safer)."""
        BATCH_SIZE = 30
        all_results = []
        
        valid_jobs = []
        for job in jobs_data:
            if job.get("description") and len(job["description"].strip()) > 0:
                valid_jobs.append(job)
        
        total_jobs = len(valid_jobs)
        
        # Process in batches
        for i in range(0, total_jobs, BATCH_SIZE):
            batch = valid_jobs[i:i+BATCH_SIZE]
            descriptions = [job["description"] for job in batch]
            
            # Get all embeddings in one API call
            embeddings = self.get_embeddings_batch(descriptions)
            
            # Store in Firestore with retry logic
            for attempt in range(3):
                try:
                    firestore_batch = self.db.batch()
                    for job, embedding in zip(batch, embeddings):
                        job_id = str(job["job_id"])
                        doc_ref = self.db.collection("vacancies").document(job_id)
                        firestore_batch.set(doc_ref, {
                            "job_id": job_id,
                            "description": job["description"],
                            "date": job.get("date"),
                            "embedding": Vector(embedding),
                            "session_id": session_id,
                            "timestamp": firestore.SERVER_TIMESTAMP
                        })
                        
                        all_results.append({
                            "job_id": job_id,
                            "embedding_dim": len(embedding)
                        })
                    
                    firestore_batch.commit()
                    break  # Success, exit retry loop
                except Exception as e:
                    if "409" in str(e) or "contention" in str(e).lower():
                        if attempt < 2:
                            print(f"Firestore contention, retrying in {2 ** attempt} seconds...")
                            time.sleep(2 ** attempt)
                        else:
                            raise
                    else:
                        raise
            
            if progress_callback:
                progress_callback(min(i + BATCH_SIZE, total_jobs), total_jobs)
            
            if i + BATCH_SIZE < total_jobs:
                time.sleep(1)
        
        return all_results
    