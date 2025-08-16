import gradio as gr
import warnings
import os, re, json, glob
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from scrape import get_jd, get_linkedin_jobs
from interview import qa_pipeline
import fitz  # PyMuPDF
import pandas as pd
import tempfile
from math import ceil
import zipfile
from typing import List, Union, Optional

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
"score": "A score out of 10, formatted as '<score>/10'."
}
"""

SUMMARIZE_JD_PROMPT = """You are an expert job description summarizer.
Given the text from a job posting page, extract and provide a concise summary of the key responsibilities and qualifications.
Return STRICT JSON only.
{
"summary": "A brief summary of the job description."
}
"""

# -----------------------------
# File helpers (fixes embedded null byte)
# -----------------------------
def load_bytes(file_input: Union[str, bytes, bytearray, dict, object]) -> bytes:
    """
    Return raw bytes from various Gradio file input types.
    Handles: filepath (str), bytes, file-like with .read(), dict with 'name'/'data'.
    """
    if file_input is None:
        raise ValueError("No file provided.")

    if isinstance(file_input, (bytes, bytearray)):
        return bytes(file_input)

    if isinstance(file_input, str):
        with open(file_input, "rb") as f:
            return f.read()

    if hasattr(file_input, "read"):
        return file_input.read()

    if isinstance(file_input, dict):
        data = file_input.get("data")
        name = file_input.get("name")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)
        if isinstance(name, str) and os.path.exists(name):
            with open(name, "rb") as f:
                return f.read()

    raise ValueError(f"Unsupported file input type: {type(file_input)}")


# -----------------------------
# PDF / Model utilities
# -----------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract text content from PDF bytes with basic validation"""
    if not pdf_bytes or len(pdf_bytes) < 5:
        raise ValueError("Uploaded file is empty or too small.")
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("Please upload a valid PDF file.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                try:
                    doc.authenticate("")  # try empty password
                except Exception:
                    raise ValueError("The PDF appears to be encrypted. Please upload an unencrypted PDF.")
            text = "\n".join([p.get_text("text") for p in doc])
    except Exception as e:
        raise ValueError(f"Unable to read PDF: {e}")

    text = re.sub(r"[ \t]+", " ", text or "").strip()
    if not text:
        raise ValueError("No extractable text found in the PDF.")
    return text


def init_model() -> ModelInference:
    """Initialize Watson X AI model"""
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return ModelInference(
        model_id=WATSONX_MODEL_ID,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID
    )


def llm_json(model: ModelInference, system_prompt: str, user_text: str, max_tokens=1000) -> dict:
    """Generate JSON response from LLM, tolerant to wrappers"""
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


# -----------------------------
# Data pipeline
# -----------------------------
def fetch_jobs(role: str, location: str, num_jobs: int, progress_callback=None) -> List[dict]:
    """Fetch jobs from LinkedIn using ScrapingDog API"""
    if progress_callback:
        progress_callback(0.3, desc="Searching for jobs...")

    pages = ceil(num_jobs / 10)  # each page returns ~10
    jobs: List[dict] = []

    for i in range(1, pages + 1):
        try:
            page_jobs = get_linkedin_jobs(role, location, SCRAPINGDOG_API_KEY, i) or []
        except Exception as e:
            page_jobs = []
            # continue but note the issue
            print(f"[fetch_jobs] page {i} error: {e}")

        jobs.extend(page_jobs)
        if len(jobs) >= num_jobs:
            jobs = jobs[:num_jobs]
            break

        if progress_callback:
            progress_callback(0.3 + (0.2 * i / max(pages, 1)), desc=f"Fetching jobs page {i}...")

    return jobs


def get_job_descriptions(jobs: List[dict], progress_callback=None) -> List[str]:
    """Get job descriptions for all jobs (summaries)"""
    if progress_callback:
        progress_callback(0.5, desc="Analyzing job descriptions...")

    summaries: List[str] = []
    for i, job in enumerate(jobs):
        jd_summary = "Job description not available"
        try:
            if job.get("job_link"):
                jd_summary = get_jd(
                    job["job_link"],
                    WATSONX_URL,
                    WATSONX_API_KEY,
                    WATSONX_PROJECT_ID,
                    WATSONX_MODEL_ID,
                    SUMMARIZE_JD_PROMPT
                ) or jd_summary
        except Exception as e:
            print(f"[get_job_descriptions] {job.get('job_link')} error: {e}")

        summaries.append(jd_summary)

        if progress_callback:
            progress_callback(
                0.5 + (0.3 * (i + 1) / max(len(jobs), 1)),
                desc=f"Processing job {i + 1}/{len(jobs)}"
            )
    return summaries


def calculate_ats_scores(jobs: List[dict], summaries: List[str], resume_text: str, model: ModelInference) -> List[dict]:
    """Calculate ATS scores for all jobs"""
    scored_jobs: List[dict] = []

    for job, jd in zip(jobs, summaries):
        ats_prompt_input = f"=== RESUME ===\n{resume_text}\n\n=== JOB DESCRIPTION ===\n{jd}"
        ats_result = llm_json(model, ATS_PROMPT, ats_prompt_input, max_tokens=120)
        score = ats_result.get("score")

        # normalize score to 'X/10' or 'Undetermined'
        norm_score = "Undetermined"
        if isinstance(score, str) and re.search(r"\d+(\.\d+)?\s*/\s*10", score):
            # clean spaces like '8 / 10' -> '8/10'
            norm_score = re.sub(r"\s*/\s*", "/", score.strip())
        elif isinstance(score, (int, float)):
            norm_score = f"{score}/10"

        sj = job.copy()
        sj["ats_score"] = norm_score
        scored_jobs.append(sj)

    return scored_jobs


def _extract_score_value(score_str: Optional[str]) -> float:
    """Helper to extract numeric value from 'X/10' else very small number for sorting"""
    if not isinstance(score_str, str):
        return -1.0
    m = re.search(r"(\d+(\.\d+)?)\s*/\s*10", score_str)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return -1.0
    return -1.0


def generate_interview_documents(jobs: List[dict], summaries: List[str], temp_dir: str) -> List[str]:
    """Generate interview preparation documents (.docx expected from qa_pipeline)"""
    cache: List[str] = []
    original_cwd = os.getcwd()

    for job, jd in zip(jobs, summaries):
        position, company = job.get("title", "Unknown Role"), job.get("company", "Unknown Company")
        job_title = f"{position} at {company}"

        if job_title in cache or jd == "Job description not available":
            continue

        os.chdir(temp_dir)
        try:
            qa_pipeline(position, company, jd)  # expected to create a .docx
            cache.append(job_title)
        except Exception as e:
            print(f"[generate_interview_documents] {job_title} error: {e}")
        finally:
            os.chdir(original_cwd)

    return cache


def create_excel_report(jobs: List[dict], temp_dir: str) -> str:
    """Create Excel report with job data (sorted by ATS score desc)"""
    df = pd.DataFrame(jobs)
    if "ats_score" not in df.columns:
        df["ats_score"] = "Undetermined"

    df_sorted = df.sort_values(
        by="ats_score",
        ascending=False,
        key=lambda s: s.map(_extract_score_value)
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

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in glob.glob(os.path.join(temp_dir, "*.docx")):
            zipf.write(file, os.path.basename(file))

    return zip_path


def create_success_message(jobs: List[dict], cache: List[str]) -> str:
    """Create success message with summary stats"""
    df = pd.DataFrame(jobs)
    if df.empty:
        return " No jobs were processed."

    if "ats_score" not in df.columns:
        df["ats_score"] = "Undetermined"

    df_sorted = df.sort_values(
        by="ats_score",
        ascending=False,
        key=lambda s: s.map(_extract_score_value)
    )

    top = df_sorted["ats_score"].iloc[0]
    bottom = df_sorted["ats_score"].iloc[-1]

    success_message = f" Successfully processed {len(jobs)} jobs!\n\n"
    success_message += f" ATS Scores Range: {top} (highest) to {bottom} (lowest)\n"
    success_message += f" Generated {len(cache)} interview preparation documents\n"
    success_message += f" Results saved to Excel spreadsheet"
    return success_message


def process_jobs(resume_file, role: str, location: str, num_jobs: int, progress=gr.Progress()):
    """Main processing pipeline"""
    try:
        progress(0.1, desc="Processing resume...")

        # ---- FIX: robust file loading (prevents embedded null byte) ----
        resume_bytes = load_bytes(resume_file)
        resume_text = extract_text_from_pdf_bytes(resume_bytes)

        progress(0.2, desc="Initializing AI model...")
        model = init_model()

        # Fetch jobs
        jobs = fetch_jobs(role, location, num_jobs, progress)
        if not jobs:
            return None, None, " No jobs found for the specified criteria."

        # Get job descriptions
        summaries = get_job_descriptions(jobs, progress)

        progress(0.8, desc="Scoring jobs and generating interview documents...")

        # Calculate ATS scores
        scored_jobs = calculate_ats_scores(jobs, summaries, resume_text, model)

        # Create temporary directory for outputs
        temp_dir = tempfile.mkdtemp(prefix="jobsearch_")

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
        # Bubble a friendly error back to the UI
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

                # ---- CHANGED: return a FILEPATH to avoid bytes-path confusion ----
                resume_file = gr.File(
                    label="Upload Resume (PDF)",
                    file_types=[".pdf"],
                    type="filepath"
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
        raise SystemExit(1)

    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=True,
        show_api=False
    )


if __name__ == "__main__":
    main()
