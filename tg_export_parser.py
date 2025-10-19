import json
import pandas as pd


def concat_text_fields(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " \n ")
    if isinstance(value, list):
        return "".join(concat_text_fields(v) for v in value)
    if isinstance(value, dict):
        # Prefer explicit 'text' key
        if "text" in value:
            return concat_text_fields(value["text"])
        # Otherwise, walk all values recursively
        parts = []
        for v in value.values():
            parts.append(concat_text_fields(v))
        return "".join(parts)
    return ""


def load_job_vacancies(uploaded_file) -> pd.DataFrame:
    """Load job vacancies from uploaded JSON file."""
    try:
        data = json.load(uploaded_file)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            candidates = data["messages"]
        elif isinstance(data, list):
            candidates = data
        else:
            candidates = []
        
        vacancies = []
        for item in candidates:
            if isinstance(item, dict) and item.get("type") == "message":
                text = concat_text_fields(item.get("text"))
                date = item.get("date")
                vacancies.append({
                    "job_id": item.get("id", len(vacancies)),
                    "description": text,
                    "date": date
                })
        
        df = pd.DataFrame(vacancies)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        raise ValueError(f"Failed to parse job vacancies JSON: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            df = load_job_vacancies(f)
            print(f"Loaded {len(df)} job vacancies")
    else:
        print("Usage: python start.py <path_to_json>")