from google.cloud import firestore
from google.cloud import aiplatform
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure
from google.cloud.firestore_v1.vector import Vector
from vertexai.generative_models import GenerativeModel
from vertexai.language_models import TextEmbeddingModel
from typing import List, Dict


class QuerySubsystem:
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        aiplatform.init(project=project_id, location=location)
        self.db = firestore.Client(project=project_id, database='ragdb')
        self.llm = GenerativeModel("gemini-2.5-flash")
        self.embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
        
    def get_top_matches(self, resume_text: str, session_id: str = None, top_k: int = 3, prompt: str = None, progress_callback=None) -> List[Dict]:
        """Find top K job matches for a resume using Firestore vector search."""
        if progress_callback:
            progress_callback("Generating resume embeddings...", 0.1)
        
        embeddings = self.embedding_model.get_embeddings([resume_text])
        resume_embedding = embeddings[0].values
        
        if progress_callback:
            progress_callback(f"Searching for top {top_k} job matches...", 0.4)
        
        query = self.db.collection("vacancies")
        if session_id:
            query = query.where("session_id", "==", session_id)
        
        vector_query = query.find_nearest(
            vector_field="embedding",
            query_vector=Vector(resume_embedding),
            distance_measure=DistanceMeasure.COSINE,
            limit=top_k,
            distance_result_field="vector_distance"
        )
        
        docs = vector_query.stream()
        top_jobs = [doc.to_dict() for doc in docs]
        print(f"Retrieved {len(top_jobs)} documents from vector search")
        
        if not top_jobs:
            print(f"No jobs found.")
            
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
    
    def delete_session_vacancies(self, session_id: str):
        """Delete all vacancies for a given session."""
        if not session_id:
            return
        
        vacancies = self.db.collection("vacancies").where("session_id", "==", session_id).stream()
        batch = self.db.batch()
        count = 0
        
        for doc in vacancies:
            batch.delete(doc.reference)
            count += 1
            if count >= 500:
                batch.commit()
                batch = self.db.batch()
                count = 0
        
        if count > 0:
            batch.commit()
        
        print(f"Deleted vacancies for session {session_id}")
