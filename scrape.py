import os, re, json, glob
from pathlib import Path
from dotenv import load_dotenv
from scrape import get_jd, get_linkedin_jobs
import fitz  
import pandas as pd

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "\n".join([p.get_text("text") for p in doc])
    return re.sub(r'[ \t]+', ' ', text).strip()

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

def find_latest_resume(directory="."):
    pdfs = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdfs:
        return None
    latest_pdf = max(pdfs, key=os.path.getmtime)
    return latest_pdf

def main():
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_URL = os.getenv("WATSONX_URL")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID")
    API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

    ATS_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst.
Given a resume and a job description, provide a concise analysis and a score.
Be completely honest with scoring so that the user gets a relaistic idea of their match with the position.
Return STRICT JSON only.
{
  "summary": "A brief summary of how well the resume matches the job description.",
  "score": "A score out of 10, formatted as '<score>/10'.",
  "missing_keywords": ["list of important keywords from the job description that are missing in the resume"]
}
"""
    SUMMARIZE_JD_PROMPT = """You are an expert job description summarizer.
    Given the text from a job posting page, extract and provide a concise summary of the key responsibilities and qualifications.
    Return STRICT JSON only.
    {
    "summary": "A brief summary of the job description."    
    }
    """ 

    resume_path = find_latest_resume()
    if not resume_path:
        print("No PDF resume found in the directory.")
        return

    with open(resume_path, "rb") as f:
        pdf_bytes = f.read()
    resume_text = extract_text_from_pdf_bytes(pdf_bytes)

    print("Initializing model...")
    model = init_model(WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID)

    role = "Software Engineering Intern"
    location = "Ireland"
    jobs = get_linkedin_jobs(role, location, API_KEY)
    summaries = []
    if jobs:
        for job in jobs:
            if job.get('job_link'):
                summaries.append(get_jd(job['job_link'], WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID, SUMMARIZE_JD_PROMPT))
        print("Fetched respective job descriptions...")
    else:
        print("No jobs found")

    print("Scoring now...")
    for i in range(len(summaries)):
        jd = summaries[i]
        ats_prompt_input = f"=== RESUME ===\n{resume_text}\n\n=== JOB DESCRIPTION ===\n{jd}"
        ats_result = llm_json(model, ATS_PROMPT, ats_prompt_input, max_tokens=500)

        summary = ats_result.get("summary", "No summary available.")
        score = ats_result.get("score", "Undetermined")
        missing = ats_result.get("missing_keywords", [])

        jobs[i]["ats_score"] = score
    df = pd.DataFrame(jobs)
    df.to_excel("jobs.xlsx")
    print("Final results saved to jobs.xlsx, good luck with the job hunt!")

if __name__ == "__main__":
    main()
