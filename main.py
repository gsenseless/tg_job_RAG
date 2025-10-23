import streamlit as st
from src.tg_export_parser import load_job_vacancies
from src.ingestion import IngestionSubsystem
from src.query import QuerySubsystem
from src.monitoring import MonitoringSubsystem
import os
import uuid

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
    
    tab1, tab2 = st.tabs(["🔍 Match Jobs", "📊 Dashboard"])
    
    with tab1:
        st.header("Job Matching")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Upload Resume (PDF)")
            resume_file = st.file_uploader("Choose your resume", type=["pdf"], key="resume")
            
            if resume_file and not st.session_state.get("resume_uploaded"):
                with st.spinner("Extracting text from resume..."):
                    try:
                        resume_text = st.session_state.ingestion.extract_text_from_pdf(resume_file)
                        st.session_state.resume_text = resume_text
                        st.success(f" Resume loaded! Text length: {len(resume_text)} chars")
                        st.session_state.resume_uploaded = True
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            st.subheader("Upload Job Vacancies (JSON)")
            jobs_file = st.file_uploader("Choose job vacancies file", type=["json"], key="jobs")
            
            if jobs_file and not st.session_state.get("jobs_uploaded"):
                with st.spinner("Loading and processing job vacancies..."):
                    try:
                        session_id = str(uuid.uuid4())
                        st.session_state.session_id = session_id
                        
                        jobs_df = load_job_vacancies(jobs_file)
                        st.info(f"Loaded {len(jobs_df)} job vacancies")
                        
                        jobs_list = jobs_df.to_dict('records')
                        st.info(f"Starting batch processing of {len(jobs_list)} jobs...")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(current, total):
                            status_text.text(f"Processing jobs {current}/{total} ...")
                            progress_bar.progress(current / total)
                        
                        try:
                            results = st.session_state.ingestion.ingest_jobs_batch(
                                jobs_list,
                                session_id,
                                progress_callback=update_progress
                            )
                        except Exception as batch_error:
                            st.error(f"Batch processing error: {batch_error}")
                            raise
                        
                        status_text.empty()
                        progress_bar.empty()
                        st.success(f" Processed {len(results)} job vacancies!")
                        st.session_state.jobs_uploaded = True
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        st.markdown("---")
        
        top_k = st.slider("Number of top matches to retrieve", min_value=1, max_value=10, value=3)
        prompt_query = st.text_area(
            "AI Reasoning Prompt", 
            value="List skills which candidate might lack for this job (if any). And list matching skills. Be concise.",
            height=100,
            help="Customize the prompt for AI-generated job matching insights"
        )
        
        if not st.session_state.get("resume_uploaded"):
            st.warning("⚠️ Please upload and process a resume first")
            return
        
        if not st.session_state.get("jobs_uploaded"):
            st.warning("⚠️ Please upload and process job vacancies first")
            return
        
        if st.button("Find Matches", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)
            
            try:
                session_id = st.session_state.get("session_id")
                resume_text = st.session_state.get("resume_text")
                matches = st.session_state.query.get_top_matches(
                    resume_text,
                    session_id,
                    top_k,
                    prompt_query,
                    progress_callback=update_progress
                )
                
                status_text.empty()
                progress_bar.empty()
                
                if matches:
                    avg_sim = sum(m["vector_distance"] for m in matches) / len(matches)
                    st.session_state.monitoring.log_query("default_user", len(matches), avg_sim)
                    st.session_state.matches = matches
                    st.success(f"Found {len(matches)} matching jobs!")
                    
                    st.session_state.query.delete_session_vacancies(session_id)
                    st.session_state.jobs_uploaded = False
                    
                    # Clean up session state except subsystems and matches
                    subsystem_keys = {"monitoring", "ingestion", "query", "matches"}
                    for key in list(st.session_state.keys()):
                        if key not in subsystem_keys:
                            del st.session_state[key]
                else:
                    st.session_state.matches = []
                    st.warning("No matches found. The vacancies collection may be empty or the vector index may not be ready.")
                    
            except Exception as e:
                status_text.empty()
                progress_bar.empty()
                st.error(f"Error message: {str(e)}")
                
        if "matches" in st.session_state and st.session_state.matches:
            for idx, match in enumerate(st.session_state.matches, 1):
                        with st.expander(f"#{idx} - Job ID: {match['job_id']} (Score: {match['vector_distance']:.3f})"):
                            st.markdown("**Job Description:**")
                            st.text_area(
                                "Full Description", 
                                value=match["description"], 
                                height=400, 
                                key=f"desc_{idx}",
                                disabled=True
                            )
                            
                            st.markdown("**🤖 AI Reasoning:**")
                            st.info(match["reasoning"])
                            
                            st.markdown("**Provide Feedback:**")
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("👍 Like", key=f"like_{idx}"):
                                    st.session_state.monitoring.log_feedback(
                                        "default_user", match["job_id"], True
                                    )
                                    st.success("Liked!")
                            with col2:
                                if st.button("👎 Dislike", key=f"dislike_{idx}"):
                                    st.session_state.monitoring.log_feedback(
                                    "default_user", match["job_id"], False
                                    )
                                    st.success("Feedback recorded!")
        
    with tab2:
        st.session_state.monitoring.render_dashboard()


if __name__ == "__main__":
    main()
