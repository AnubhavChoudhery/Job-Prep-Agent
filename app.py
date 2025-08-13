import os, re, json, glob
from pathlib import Path
from dotenv import load_dotenv
import fitz  # PyMuPDF

# watsonx.ai
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames

# ---------- env ----------
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID")

# ---------- Job Description ----------
# PASTE YOUR JOB DESCRIPTION HERE
JOB_DESCRIPTION = """
Qualifications:
Pursuing a bachelor's or master's degree in computer science, engineering, or another related field. Must graduate before July 2027.

Previous internship experience.

Working towards a proficiency of one or more programming languages such as Typescript, Node.js, or Python.

You find large challenges exciting and enjoy discovering problems as much as solving them.

You are able to problem-solve and adapt to changing priorities in a fast-paced, dynamic environment.

This internship will take place from May - September (based on your summer schedule) and you will need to be able to work out of our NY or SF office during this time.

Skills You'll Need To Bring:
Thoughtful problem-solving: For you, problem-solving starts with a clear and accurate understanding of the context. You can decompose tricky problems and work towards a clean solution, by yourself or with teammates. You're comfortable asking for help when you get stuck.

AI enthusiast: You have built or prototyped features with AI technologies (LLMs, Embeddings, ML)

Put users first: You think critically about the implications of what you're building, and how it shapes real people's lives. You understand that reach comes with responsibility for our impact—good and bad.

Not ideological about technology: To you, technologies and programming languages are about tradeoffs. You may be opinionated, but you're not ideological and can learn new technologies as you go.

Empathetic communication: You communicate nuanced ideas clearly, whether you're explaining technical decisions in writing or brainstorming in real time. In disagreements, you engage thoughtfully with other perspectives and compromise when needed.

Team player: For you, work isn't a solo endeavor. You enjoy collaborating cross-functionally to accomplish shared goals, and you care about learning, growing, and helping others to do the same.

Nice to Haves:
You have expertise with specific technologies that are part of our stack, including Typescript, React, Python.

You've heard of computing pioneers like Ada Lovelace, Douglas Engelbart, Alan Kay, and others—and understand why we're big fans of their work.

You have interests outside of technology, such as in art, history, or social sciences.
"""

# ---------- prompts ----------
ATS_PROMPT = """You are an expert ATS (Applicant Tracking System) analyst.
Given a resume and a job description, provide a concise analysis and a score.
Return STRICT JSON only.
{
  "summary": "A brief summary of how well the resume matches the job description.",
  "score": "A score out of 10, formatted as '<score>/10'.",
  "missing_keywords": ["list of important keywords from the job description that are missing in the resume"]
}
"""

# ---------- helpers ----------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        text = "\n".join([p.get_text("text") for p in doc])
    return re.sub(r'[ \t]+', ' ', text).strip()

def init_model() -> ModelInference:
    missing = []
    if not WATSONX_API_KEY:    missing.append("WATSONX_API_KEY")
    if not WATSONX_URL:        missing.append("WATSONX_URL")
    if not WATSONX_PROJECT_ID: missing.append("WATSONX_PROJECT_ID")
    if not WATSONX_MODEL_ID:   missing.append("WATSONX_MODEL_ID")
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

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
    print("Finding the latest resume...")
    resume_path = find_latest_resume()
    if not resume_path:
        print("No PDF resume found in the directory.")
        return

    print(f"Found resume: {resume_path}")

    with open(resume_path, "rb") as f:
        pdf_bytes = f.read()

    resume_text = extract_text_from_pdf_bytes(pdf_bytes)
    jd_text = JOB_DESCRIPTION.strip()

    print("Initializing model...")
    model = init_model()

    print("Analyzing resume against job description...")
    ats_prompt_input = f"=== RESUME ===\n{resume_text}\n\n=== JOB DESCRIPTION ===\n{jd_text}"
    ats_result = llm_json(model, ATS_PROMPT, ats_prompt_input, max_tokens=500)

    summary = ats_result.get("summary", "No summary available.")
    score_val = ats_result.get("score", "0/10")
    missing = ats_result.get("missing_keywords", [])

    print("\n--- ATS Analysis ---")
    print(f"Score: {score_val}")
    print(f"Summary: {summary}")
    if missing:
        print("Missing Keywords:")
        for keyword in missing:
            print(f"- {keyword}")
    print("--------------------")

if __name__ == "__main__":
    main()
