# Job-Resume RAG Matching System

AI-powered job matching system that uses Vertex AI embeddings and LLM reasoning to match resumes with job vacancies.

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
export GCP_PROJECT_ID="your-project-id"
export GCP_LOCATION="us-central1"

# Place your credentials.json in the project root
docker compose up -d
```

Access the application at http://localhost:8501

#### Option B: Local Python

```bash
streamlit run main.py
```

## Usage

### 1. Ingestion Tab

**Upload Resume:**
- Optionally specify user ID

**Upload Job Vacancies:**
- Upload JSON file with job vacancies (Telegram export format)

### 2. Query & Match Tab

- Select number of top matches 
- Review AI-generated match explanations
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

- **UI**: Streamlit
- **Embeddings**: Vertex AI Text Embedding
- **Database**: Google Cloud Firestore
- **PDF Processing**: pdfplumber
- **Similarity**: scikit-learn cosine similarity
- **Visualization**: Plotly

*Note that processing is quite slow due to LLM quotas limits.



