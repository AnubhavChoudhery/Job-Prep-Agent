import requests
import os
import json
from bs4 import BeautifulSoup
from pathlib import Path
import re
from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
import warnings

warnings.filterwarnings("ignore")
def init_model(WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID) -> ModelInference:
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return ModelInference(
        model_id=WATSONX_MODEL_ID,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID
    )

def llm_json(model: ModelInference, system_prompt: str, user_text: str, max_tokens=1000):
    params = {
        GenTextParamsMetaNames.MAX_NEW_TOKENS: max_tokens,
        GenTextParamsMetaNames.TEMPERATURE: 0.2,
        GenTextParamsMetaNames.DECODING_METHOD: "greedy",
    }
    prompt = f"{system_prompt}\n\n=== INPUT START ===\n{user_text}\n=== INPUT END ===="

    out = model.generate_text(prompt=prompt, params=params)

    if isinstance(out, str):
        raw = out.strip()
    elif isinstance(out, dict):
        raw = out.get("results", [{}])[0].get("generated_text", "").strip()
    else:
        raw = str(out).strip()

    try:
        fb = raw.find("{"); lb = raw.rfind("}")
        return json.loads(raw[fb:lb+1]) if fb != -1 and lb != -1 else {}
    except Exception:
        return {}

def get_linkedin_jobs(role, location, API_KEY, page):
    url = "https://api.scrapingdog.com/linkedinjobs"
    params = {
        "api_key": API_KEY,
        "field": role,
        "location": location,
        "page": page
    }
    
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        jobs_data = r.json()
        
        results = []
        for job in jobs_data:
            results.append({
                "title": job.get("job_position"),
                "company": job.get("company_name"),
                "location": job.get("job_location"),
                "job_link": job.get("job_link"),
                "job_id": job.get("job_id"),
                "posted_date": job.get("job_posting_date"),
                "ats_score": "Undetermined"
            })
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_jd(url, WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID, SUMMARIZE_JD_PROMPT):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    job_text = soup.get_text()
    job_text = re.sub(r'[ \t]+', ' ', job_text).strip()
    model = init_model(WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID)
    summary_result = llm_json(model, SUMMARIZE_JD_PROMPT, job_text, max_tokens=2000)
    
    summary = summary_result.get("summary", "Could not generate summary.")
    return summary

if __name__ == "__main__":
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_URL = os.getenv("WATSONX_URL")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID")
    API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

    SUMMARIZE_JD_PROMPT = """You are an expert job description summarizer.
    Given the text from a job posting page, extract and provide a concise summary of the key responsibilities and qualifications.
    Return STRICT JSON only.
    {
    "summary": "A brief summary of the job description."    
    }
    """ 

    role = "Software Engineering Intern"
    location = "USA"
    pages = 3
    jobs = []
    for i in range(1, pages+1):
        jobs += get_linkedin_jobs(role, location, API_KEY, i)
    if jobs:
        summaries = []
        for job in jobs:
            if job.get('job_link'):
                summaries.append(get_jd(job['job_link'], WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID, SUMMARIZE_JD_PROMPT))
        print(len(jobs))
    else:
        print("No jobs found")
