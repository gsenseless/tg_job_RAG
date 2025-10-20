from google.cloud import firestore
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple


class QuerySubsystem:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
        self.db = firestore.Client(project=project_id, database='ragdb')
        self.llm = GenerativeModel("gemini-2.5-flash")
        
    def get_resume(self, user_id: str = "default_user") -> Dict:
        """Retrieve resume from Firestore."""
        doc = self.db.collection("resumes").document(user_id).get()
        if not doc.exists:
            raise ValueError(f"Resume not found for user {user_id}")
        return doc.to_dict()
    
    def get_all_jobs(self) -> List[Dict]:
        """Retrieve all job vacancies from Firestore."""
        docs = self.db.collection("jobs").stream()
        return [doc.to_dict() for doc in docs]
    
    def compute_similarities(self, resume_embedding: List[float], job_embeddings: List[List[float]]) -> np.ndarray:
        """Compute cosine similarity between resume and job embeddings."""
        resume_emb = np.array(resume_embedding).reshape(1, -1)
        jobs_emb = np.array(job_embeddings)
        return cosine_similarity(resume_emb, jobs_emb)[0]
    
    def get_top_matches(self, user_id: str = "default_user", top_k: int = 3, prompt: str = None, progress_callback=None) -> List[Dict]:
        """Find top K job matches for a resume."""
        if progress_callback:
            progress_callback("Fetching resume...", 0.1)
        
        resume = self.get_resume(user_id)
        resume_embedding = resume["embedding"]
        resume_text = resume["text"]
        
        if progress_callback:
            progress_callback("Fetching job vacancies...", 0.2)
        
        jobs = self.get_all_jobs()
        if not jobs:
            return []
        
        if progress_callback:
            progress_callback(f"Computing similarities for {len(jobs)} jobs...", 0.4)
        
        job_embeddings = [job["embedding"] for job in jobs]
        similarities = self.compute_similarities(resume_embedding, job_embeddings)
        
        for idx, job in enumerate(jobs):
            job["similarity_score"] = float(similarities[idx])
        
        jobs_sorted = sorted(jobs, key=lambda x: x["similarity_score"], reverse=True)
        top_jobs = jobs_sorted[:top_k]
        
        if progress_callback:
            progress_callback(f"Generating AI insights for top {top_k} matches...", 0.6)
        
        for idx, job in enumerate(top_jobs):
            if progress_callback:
                progress_pct = 0.6 + (0.4 * (idx + 1) / len(top_jobs))
                progress_callback(f"Generating insight {idx + 1}/{len(top_jobs)}...", progress_pct)
            job["reasoning"] = self.generate_reasoning(resume_text, job["description"], prompt)
        
        return top_jobs
    
    def generate_reasoning(self, resume_text: str, job_description: str, custom_prompt: str = None) -> str:
        if custom_prompt is None:
            custom_prompt = "List skills which candidate might lack for this job (if any). And list matching skills."
        
        prompt = f"""{custom_prompt}
Resume:
{resume_text[:3000]}

Job Description:
{job_description[:3000]}

"""
        
        response = self.llm.generate_content(prompt)
        return response.text
