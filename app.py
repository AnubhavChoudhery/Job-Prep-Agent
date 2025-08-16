import gradio as gr
import warnings
import os, re, json, glob
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from scrape import get_jd, get_linkedin_jobs
from interview import qa_pipeline
import fitz  
import pandas as pd
import tempfile
from math import ceil
import zipfile
from typing import List

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

warnings.filterwarnings("ignore")

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID")
SCRAPINGDOG_API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

ATS_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst.
Given a resume and a job description, provide a concise analysis and a score.
Be completely honest with scoring so that the user gets a realistic idea of their match with the position.
Return STRICT JSON only.
{
"score": "A score out of 10, formatted as '<score>/10'.",
}
"""

SUMMARIZE_JD_PROMPT = """You are an expert job description summarizer.
Given the text from a job posting page, extract and provide a concise summary of the key responsibilities and qualifications.
Return STRICT JSON only.
{
"summary": "A brief summary of the job description."    
}
""" 

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes"""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "\n".join([p.get_text("text") for p in doc])
    return re.sub(r'[ \t]+', ' ', text).strip()

def init_model() -> ModelInference:
    """Initialize Watson X AI model"""
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return ModelInference(
        model_id=WATSONX_MODEL_ID,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID
    )

def llm_json(model: ModelInference, system_prompt: str, user_text: str, max_tokens=1000):
    """Generate JSON response from LLM"""
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

def fetch_jobs(role: str, location: str, num_jobs: int, progress_callback=None) -> List[dict]:
    """Fetch jobs from LinkedIn using ScrapingDog API"""
    if progress_callback:
        progress_callback(0.3, desc="Searching for jobs...")
    
    pages = ceil(num_jobs/10)*10
    
    jobs = []
    for i in range(1, pages + 1):
        page_jobs = get_linkedin_jobs(role, location, SCRAPINGDOG_API_KEY, i)
        jobs.extend(page_jobs)
        if len(jobs) >= num_jobs:
            jobs = jobs[:num_jobs]  # Limit to requested number
            break
        if progress_callback:
            progress_callback(0.3 + (0.2 * i / pages), desc=f"Fetching jobs page {i}...")
    
    return jobs

def get_job_descriptions(jobs: List[dict], progress_callback=None) -> List[str]:
    """Get job descriptions for all jobs"""
    if progress_callback:
        progress_callback(0.5, desc="Analyzing job descriptions...")
    
    summaries = []
    for i, job in enumerate(jobs):
        if job.get('job_link'):
            jd_summary = get_jd(
                job['job_link'], 
                WATSONX_URL, 
                WATSONX_API_KEY, 
                WATSONX_PROJECT_ID, 
                WATSONX_MODEL_ID, 
                SUMMARIZE_JD_PROMPT
            )
            summaries.append(jd_summary)
        else:
            summaries.append("Job description not available")
        
        if progress_callback:
            progress_callback(
                0.5 + (0.3 * (i + 1) / len(jobs)), 
                desc=f"Processing job {i+1}/{len(jobs)}"
            )
    
    return summaries

def calculate_ats_scores(jobs: List[dict], summaries: List[str], resume_text: str, model: ModelInference) -> List[dict]:
    """Calculate ATS scores for all jobs"""
    scored_jobs = []
    
    for i, (job, jd) in enumerate(zip(jobs, summaries)):
        ats_prompt_input = f"=== RESUME ===\n{resume_text}\n\n=== JOB DESCRIPTION ===\n{jd}"
        ats_result = llm_json(model, ATS_PROMPT, ats_prompt_input, max_tokens=100)
        score = ats_result.get("score", "Undetermined")
        
        # Create a copy of job with score
        scored_job = job.copy()
        scored_job["ats_score"] = score
        scored_jobs.append(scored_job)
    
    return scored_jobs

def generate_interview_documents(jobs: List[dict], summaries: List[str], temp_dir: str) -> List[str]:
    """Generate interview preparation documents"""
    cache = []
    original_cwd = os.getcwd()
    
    for job, jd in zip(jobs, summaries):
        position, company = job["title"], job["company"]
        job_title = f"{position} at {company}"
        
        if job_title not in cache and jd != "Job description not available":
            os.chdir(temp_dir)
            try:
                qa_pipeline(position, company, jd)
                cache.append(job_title)
            except Exception as e:
                print(f"Error generating interview document for {job_title}: {e}")
            finally:
                os.chdir(original_cwd)
    
    return cache

def create_excel_report(jobs: List[dict], temp_dir: str) -> str:
    """Create Excel report with job data"""
    df = pd.DataFrame(jobs)
    df_sorted = df.sort_values(
        by="ats_score", 
        ascending=False, 
        key=lambda x: x.str.extract('(\d+)')[0].astype(float)
    )
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    excel_filename = f"jobs_{timestamp}.xlsx"
    excel_path = os.path.join(temp_dir, excel_filename)
    df_sorted.to_excel(excel_path, index=False)
    
    return excel_path

def create_documents_zip(temp_dir: str) -> str:
    """Create ZIP file with all interview documents"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    zip_filename = f"interview_documents_{timestamp}.zip"
    zip_path = os.path.join(temp_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in glob.glob(os.path.join(temp_dir, "*.docx")):
            zipf.write(file, os.path.basename(file))
    
    return zip_path

def create_success_message(jobs: List[dict], cache: List[str]) -> str:
    """Create success message with summary stats"""
    df = pd.DataFrame(jobs)
    df_sorted = df.sort_values(
        by="ats_score", 
        ascending=False, 
        key=lambda x: x.str.extract('(\d+)')[0].astype(float)
    )
    
    success_message = f" Successfully processed {len(jobs)} jobs!\n\n"
    success_message += f" ATS Scores Range: {df_sorted['ats_score'].iloc[0]} (highest) to {df_sorted['ats_score'].iloc[-1]} (lowest)\n"
    success_message += f" Generated {len(cache)} interview preparation documents\n"
    success_message += f" Results saved to Excel spreadsheet"
    
    return success_message

def process_jobs(resume_file, role: str, location: str, num_jobs: int, progress=gr.Progress()):
    """Main processing pipeline using functional approach"""
    try:
        progress(0.1, desc="Processing resume...")
        
        # Extract text from uploaded PDF
        resume_bytes = resume_file.read() if hasattr(resume_file, 'read') else open(resume_file, 'rb').read()
        resume_text = extract_text_from_pdf_bytes(resume_bytes)
        
        progress(0.2, desc="Initializing AI model...")
        model = init_model()
        
        # Fetch jobs
        jobs = fetch_jobs(role, location, num_jobs, progress)
        
        if not jobs:
            return None, None, "No jobs found for the specified criteria."
        
        # Get job descriptions
        summaries = get_job_descriptions(jobs, progress)
        
        progress(0.8, desc="Scoring jobs and generating interview documents...")
        
        # Calculate ATS scores
        scored_jobs = calculate_ats_scores(jobs, summaries, resume_text, model)
        
        # Create temporary directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        # Generate interview documents
        cache = generate_interview_documents(scored_jobs, summaries, temp_dir)
        
        progress(0.9, desc="Creating final outputs...")
        
        # Create Excel report
        excel_path = create_excel_report(scored_jobs, temp_dir)
        
        # Create ZIP with documents
        zip_path = create_documents_zip(temp_dir)
        
        # Create success message
        success_message = create_success_message(scored_jobs, cache)
        
        progress(1.0, desc="Complete!")
        
        return excel_path, zip_path, success_message
        
    except Exception as e:
        return None, None, f" Error processing jobs: {str(e)}"

def process_and_update(resume_file, role_input, location_input, num_jobs_input):
    """Process jobs and update UI components"""
    progress_bar = gr.Progress()
    excel_file, zip_file, message = process_jobs(
        resume_file, role_input, location_input, num_jobs_input, progress=progress_bar
    )
    
    # Update visibility based on success
    excel_visible = excel_file is not None
    zip_visible = zip_file is not None
    
    return (
        message,  
        gr.update(value=excel_file, visible=excel_visible),  
        gr.update(value=zip_file, visible=zip_visible)  
    )

def create_interface():
    """Create and configure Gradio interface"""
    with gr.Blocks(title="Job Search Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # AI-Powered Job Search Assistant
        
        Upload your resume, specify your target role and location, and get AI-powered job matching with interview preparation documents!
        
        ## How it works:
        1. **Upload** your resume (PDF format)
        2. **Specify** the role you're looking for and location
        3. **Set** the number of jobs to analyze
        4. **Get** ATS scores, ranked job matches, and interview preparation documents
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input Parameters")
                
                resume_file = gr.File(
                    label="Upload Resume (PDF)",
                    file_types=[".pdf"],
                    type="binary"
                )
                
                role_input = gr.Textbox(
                    label="Job Role",
                    placeholder="e.g., Software Engineer, Data Scientist, Product Manager",
                    value="Software Engineer"
                )
                
                location_input = gr.Textbox(
                    label="Location",
                    placeholder="e.g., USA, Canada, Ireland, Remote",
                    value="USA"
                )
                
                num_jobs_input = gr.Slider(
                    minimum=5,
                    maximum=500,
                    step=5,
                    value=15,
                    label="Number of Jobs to Analyze"
                )
                
                process_btn = gr.Button(" Start Job Search & Analysis", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("###  Results & Downloads")
                
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Results will appear here...",
                    lines=6,
                    interactive=False
                )
                
                excel_download = gr.File(
                    label="Download Excel Report",
                    visible=False
                )
                
                documents_download = gr.File(
                    label=" Download Interview Documents (ZIP)",
                    visible=False
                )
        
        # Connect button to processing function
        process_btn.click(
            fn=process_and_update,
            inputs=[resume_file, role_input, location_input, num_jobs_input],
            outputs=[status_output, excel_download, documents_download]
        )
        
        # Footer
        gr.Markdown("""
        ---
        ###  Tips for Best Results:
        - Ensure your resume is a clear, well-formatted PDF
        - Use specific job titles for better matching
        - Try different locations to expand your search
        - The system will generate interview prep documents for unique company-role combinations
        
        ### Features:
        - **ATS Scoring**: Get realistic compatibility scores for each job
        - **Smart Ranking**: Jobs sorted by ATS score (best matches first)
        - **Interview Prep**: Customized documents with company-specific interview insights
        - **Export Ready**: Excel spreadsheet with all job details and scores
        """)
    
    return app

def check_environment_variables() -> bool:
    """Check if all required environment variables are set"""
    required_env_vars = [
        "WATSONX_API_KEY", "WATSONX_URL", "WATSONX_PROJECT_ID", 
        "WATSONX_MODEL_ID", "SCRAPINGDOG_API_KEY"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        return False
    return True

def main():
    """Main function to run the application"""
    if not check_environment_variables():
        exit(1)
    
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True,  
        show_api=False
    )

if __name__ == "__main__":
    main()
