import streamlit as st
from tg_export_parser import load_job_vacancies
from ingestion import IngestionSubsystem
from query import QuerySubsystem
from monitoring import MonitoringSubsystem
import os
import random

st.set_page_config(page_title="Job-Resume Matcher", page_icon="💼", layout="wide")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION")


def main():
    st.title("Job-Resume Matching System")
    st.markdown("Upload your resume and job vacancies to find the best matches using AI")
    
    if "ingestion" not in st.session_state:
        st.session_state.ingestion = IngestionSubsystem(PROJECT_ID, LOCATION)
    if "query" not in st.session_state:
        st.session_state.query = QuerySubsystem(PROJECT_ID, LOCATION)
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = MonitoringSubsystem(PROJECT_ID)
    if "random_user_id" not in st.session_state:
        names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
        st.session_state.random_user_id = random.choice(names).lower()
    
    tab1, tab2, tab3 = st.tabs(["📤 Ingestion", "🔍 Query & Match", "📊 Dashboard"])
    
    with tab1:
        st.header("1. Data Ingestion")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Resume (PDF)")
            resume_file = st.file_uploader("Choose your resume", type=["pdf"], key="resume")
            user_id = st.text_input("User ID (optional)", value=st.session_state.random_user_id)
            
            if st.button("Process Resume", type="primary"):
                if resume_file:
                    with st.spinner("Extracting text and generating embeddings..."):
                        try:
                            result = st.session_state.ingestion.ingest_resume(resume_file, user_id)
                            st.success(f" Resume processed! Text length: {result['text_length']} chars")
                            st.session_state.resume_uploaded = True
                            if st.session_state.get("jobs_uploaded"):
                                st.info(" Both ingestion steps complete! Go to the **Query & Match** tab to find matching jobs.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please upload a resume file")
        
        with col2:
            st.subheader("Upload Job Vacancies (JSON)")
            jobs_file = st.file_uploader("Choose job vacancies file", type=["json"], key="jobs")
            
            if st.button("Process Job Vacancies", type="primary"):
                if jobs_file:
                    with st.spinner("Loading and processing job vacancies..."):
                        try:
                            jobs_df = load_job_vacancies(jobs_file)
                            st.info(f"Loaded {len(jobs_df)} job vacancies")
                            
                            # Convert DataFrame to list of dicts
                            jobs_list = jobs_df.to_dict('records')
                            st.info(f"Starting batch processing of {len(jobs_list)} jobs...")
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            def update_progress(current, total):
                                status_text.text(f"Processing jobs {current}/{total} (batches of 250)...")
                                progress_bar.progress(current / total)
                            
                            try:
                                results = st.session_state.ingestion.ingest_jobs_batch(
                                    jobs_list, 
                                    progress_callback=update_progress
                                )
                            except Exception as batch_error:
                                st.error(f"Batch processing error: {batch_error}")
                                raise
                            
                            status_text.empty()
                            progress_bar.empty()
                            st.success(f" Processed {len(results)} job vacancies!")
                            st.session_state.jobs_uploaded = True
                            if st.session_state.get("resume_uploaded"):
                                st.info(" Both ingestion steps complete! Go to the **Query & Match** tab to find matching jobs.")
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.warning("Please upload a job vacancies file")
    
    with tab2:
        st.header("2. Find Job Matches")
        
        if not st.session_state.get("resume_uploaded"):
            st.warning("⚠️ Please upload and process a resume first in the Ingestion tab")
            return
        
        if not st.session_state.get("jobs_uploaded"):
            st.warning("⚠️ Please upload and process job vacancies first in the Ingestion tab")
            return
        
        user_id_query = st.text_input("User ID", value=st.session_state.random_user_id, key="user_query")
        top_k = st.slider("Number of top matches to retrieve", min_value=1, max_value=20, value=10)
        
        if st.button("🔍 Find Matches", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)
            
            try:
                matches = st.session_state.query.get_top_matches(
                    user_id_query, 
                    top_k,
                    progress_callback=update_progress
                )
                
                status_text.empty()
                progress_bar.empty()
                
                if matches:
                    avg_sim = sum(m["similarity_score"] for m in matches) / len(matches)
                    st.session_state.monitoring.log_query(user_id_query, len(matches), avg_sim)
                    st.session_state.matches = matches
                    st.success(f"Found {len(matches)} matching jobs!")
                else:
                    st.session_state.matches = []
                    
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Error: {e}")
        
        if "matches" in st.session_state and st.session_state.matches:
            for idx, match in enumerate(st.session_state.matches, 1):
                        with st.expander(f"#{idx} - Job ID: {match['job_id']} (Score: {match['similarity_score']:.3f})"):
                            st.markdown("**Job Description:**")
                            st.write(match["description"][:500] + "..." if len(match["description"]) > 500 else match["description"])
                            
                            st.markdown("**🤖 AI Reasoning:**")
                            st.info(match["reasoning"])
                            
                            st.markdown("**Provide Feedback:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👍 Like", key=f"like_{idx}"):
                                    st.session_state.monitoring.log_feedback(
                                        user_id_query, match["job_id"], True
                                    )
                                    st.success("Liked!")
                            with col2:
                                if st.button("👎 Dislike", key=f"dislike_{idx}"):
                                    st.session_state.monitoring.log_feedback(
                                        user_id_query, match["job_id"], False
                                    )
                                    st.success("Feedback recorded!")
    
    with tab3:
        st.session_state.monitoring.render_dashboard()


if __name__ == "__main__":
    main()
