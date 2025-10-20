# Job-Resume RAG Matching System

A large number of job vacancies are posted in various Telegram channels, often aimed at a wide audience. As a result, 90% of these listings are typically not relevant to a specific job seeker, because most people have a unique skill set and background. 

**This project is designed to solve that problem.**  
With Job-Resume RAG Matching System, we use Retrieval Augmented Generation (RAG) approaches to intelligently match job postings with user resumes, surfacing only the most relevant opportunities based on each individual’s skills and experiences. This helps job seekers avoid wasting time on irrelevant listings and increases the chances of finding suitable positions faster.

## System Architecture

```
┌─────────────────┐
│  Streamlit UI   │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
┌───▼──┐  ┌──▼────┐
│Ingest│  │Query  │
│      │  │       │
└───┬──┘  └──┬────┘
    │        │
    ▼        ▼
┌──────────────────┐
│   Vertex AI      │
│   - Embeddings   │
│   - Gemini LLM   │
└─────────┬────────┘
          │
     ┌────▼─────┐
     │Firestore │
     │  - Jobs  │
     │  - Resume│
     │  - Logs  │
     └──────────┘

## How it works?

1. **Ingestion:**  
   - The user's resume and a JSON containing job vacancies are uploaded into the system.
   - Both the resume and each job posting are converted into embeddings (vector representations) using Vertex AI models.
   - These embeddings, along with the original text data, are stored in Google Firestore.

2. **Matching:**  
   - When a match is requested, the system computes vector similarities between the resume’s embedding and all job vacancy embeddings in Firestore.
   - The top matches (most relevant job postings) are retrieved based on these similarity scores.

3. **Augmented Generation:**  
   - For each top-matched job posting, a prompt is constructed combining the user's skills, experience, and the job description.
   - Vertex AI’s Gemini LLM generates a tailored summary or explanation for why this job is a good match for the candidate, helping the applicant quickly understand their fit.

_In summary: The system ingests data, finds top matches using vector search, and augments results with contextual AI-generated insights for personalized job recommendations._


```

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Configure GCP Credentials

#### Create Service Account and Generate JSON Credentials

1. **Go to Google Cloud Console**:
   - Navigate to [IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts)

2. **Create Service Account**:

3. **Grant Required Roles**:
   - **Vertex AI User** (`roles/aiplatform.user`) - For embeddings and LLM
   - **Cloud Datastore User** (`roles/datastore.user`) - For Firestore read/write
   - **Cloud Datastore Index Admin** (`roles/datastore.indexAdmin`) - For Firestore queries
   - **Service Account Token Creator** (`roles/iam.serviceAccountTokenCreator`) - For authentication

4. **Generate JSON Key**:
   - Click on the created service account
   - Go to "Keys" tab
   - Click "Add Key" > "Create new key"
   - Select "JSON" format
   - Download the JSON file (e.g., `your-project-credentials.json`)

5. **Set Environment Variables**:

```bash
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-project-credentials.json"
```

### 3. Enable Required APIs

Enable these Google Cloud APIs in your project:

1. **Vertex AI API**:
   enable via [Console](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com)

2. **Cloud Firestore API**:
   enable via [Console](https://console.cloud.google.com/apis/library/firestore.googleapis.com)
   - Create Firestore database in Native mode if not exists

3. **Cloud Storage API** (optional, but recommended):
   enable via [Console](https://console.cloud.google.com/apis/library/storage.googleapis.com)

### 4. Run the Application

#### Option A: Docker

```bash
# Set environment variables
export PORT=8501  # Optional, defaults to 8501. Might be useful for cloud deployment.

# Place your credentials.json in the project root
docker compose up -d
```

#### Option B: Local Python

```bash
streamlit run main.py
```

## Usage
Access the application at http://localhost:8501 (or your configured PORT)

### 1. Ingestion Tab

**Upload Resume:**
- Optionally specify user ID

**Upload Job Vacancies:**
- Upload JSON file with job vacancies (Telegram export format)
https://telegram.org/blog/export-and-more

### 2. Query & Match Tab

- Select number of top matches
- Adjust prompt
- Review AI-generated reasoning
- Provide feedback on each match

### 3. Dashboard Tab
View 6 interactive charts.

## Data Format

### Job Vacancies JSON Format

```json
{
  "messages": [
    {
      "id": 1,
      "type": "message",
      "date": "2024-01-01T12:00:00",
      "text": "Job description text here..."
    }
  ]
}
```

## Technology Stack

- **User Interface**: Streamlit
- **Embeddings and LLM**: Vertex AI Text Embedding and Gemini LLM.
- **Database (knowledge base)**: Google Cloud Firestore
- **PDF Processing**: pdfplumber
- **Similarity**: scikit-learn cosine similarity
- **Visualization, Dashboard**: Plotly, Streamlit
- **Containerization**: docker-compose

*Note that processing is quite slow due to LLM quotas limits.



