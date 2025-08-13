import requests
import os
from app import init_model, llm_json

from bs4 import BeautifulSoup
from pathlib import Path
import re
from dotenv import load_dotenv

def get_linkedin_jobs(role, location):
    url = "https://api.scrapingdog.com/linkedinjobs"
    params = {
        "api_key": API_KEY,
        "field": role,
        "location": location,
        "page": 1
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
                "posted_date": job.get("job_posting_date")
            })
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def get_jd(url: str):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    job_text = soup.get_text()
    job_text = re.sub(r'[ \t]+', ' ', job_text).strip()
    model = init_model()
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
    location = "Ireland"
    jobs = get_linkedin_jobs(role, location)
    
    if jobs:
        summaries = []
        for job in jobs:
            if job.get('job_link'):
                summaries.append(get_jd(job['job_link']))
        print(summaries)
    else:
        print("No jobs found")
