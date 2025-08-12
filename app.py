import os, re, json
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import fitz  # PyMuPDF
from rapidfuzz import fuzz, process
from spreadsheet import SpreadsheetAgent

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

MAX_PDF_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_MIME = {"application/pdf"}

# ---------- prompts ----------
RESUME_PROMPT = """You are a skill-extraction agent for ANY profession.
Return STRICT JSON only.
{
  "skills": ["string"],
  "years_experience_estimate": 0.0,
  "seniority_indicators": ["string"],
  "meta": { "has_contact": true, "has_sections": true }
}"""

JD_PROMPT = """You are a job-requirement extraction agent.
Return STRICT JSON only.
{
  "required_skills": ["string"],
  "priority": {"skill": 1},
  "seniority_required": "intern|junior|mid|senior|lead|manager|director|vp|cxo",
  "role_title": "string"
}"""

app = Flask(__name__, template_folder="templates")

# ---------- helpers ----------
def is_pdf(file_storage) -> bool:
    ctype = (file_storage.mimetype or "").lower()
    if ctype in ALLOWED_MIME:
        return True
    head = file_storage.stream.read(5)
    file_storage.stream.seek(0)
    return head == b"%PDF-"

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
    prompt = f"{system_prompt}\n\n=== INPUT START ===\n{user_text}\n=== INPUT END ==="

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

def normalize(s: str) -> str:
    s = s.lower().strip()
    return re.sub(r'[^a-z0-9+\-#./ ]', '', s)

def seniority_bucket(indicators):
    joined = " ".join(indicators).lower()
    if any(k in joined for k in ["director","head","vp","chief","cxo"]): return "director"
    if any(k in joined for k in ["lead","manager"]): return "manager"
    if "senior" in joined: return "senior"
    if any(k in joined for k in ["intern","trainee"]): return "intern"
    return "mid"

def fuzzy_coverage(resume_skills, jd_skills, cutoff=85):
    r = [normalize(x) for x in resume_skills]
    j = [normalize(x) for x in jd_skills]
    matches, gaps = [], []
    for js in j:
        found = process.extractOne(js, r, scorer=fuzz.token_set_ratio)
        if found is None:
            gaps.append(js); continue
        match, score, _ = found
        if score >= cutoff: matches.append((js, match, score))
        else: gaps.append(js)
    return matches, gaps

def score_ats(matches, gaps, jd_priority, resume_meta, resume_sen, jd_sen):
    ladder = ["intern","junior","mid","senior","manager","director","vp","cxo"]
    idx = lambda x: ladder.index(x) if x in ladder else 2

    all_needed = set(list(jd_priority.keys()) + gaps)
    total_w = sum(jd_priority.get(k,1) for k in all_needed) or 1
    matched_w = sum(jd_priority.get(m[0],1) for m in matches)
    coverage_pct = matched_w / total_w
    coverage_pts = 6.0 * coverage_pct

    diff = abs(idx(resume_sen) - idx(jd_sen))
    seniority_pts = max(0.0, 2.0 - 0.7 * diff)

    fmt_pts = (1.0 if resume_meta.get("has_contact") else 0.0) + (1.0 if resume_meta.get("has_sections") else 0.0)

    score10 = round(min(10.0, coverage_pts + seniority_pts + fmt_pts), 2)
    return score10

def get_sheet_agent() -> SpreadsheetAgent:
    mode = os.getenv("LOG_MODE", "csv").lower()
    if mode == "gsheets":
        return SpreadsheetAgent(
            mode="gsheets",
            gsa_json_path=os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"),
            gsheet_id=os.getenv("GOOGLE_SHEET_ID"),
            worksheet_name=os.getenv("GOOGLE_WORKSHEET", "Sheet1"),
            debug=True,
        )
    return SpreadsheetAgent(
        mode="csv",
        csv_path=os.getenv("CSV_LOG_PATH", "logs/applications.csv"),
        debug=True,
    )

# ---------- routes ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

@app.route("/score", methods=["POST"])
def score():
    if "resume_pdf" not in request.files or "jd_text" not in request.form:
        return "resume_pdf file and jd_text are required", 400

    file = request.files["resume_pdf"]
    if not is_pdf(file):
        return "Please upload a valid PDF file", 400

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_PDF_BYTES:
        return f"PDF too large. Max {MAX_PDF_BYTES//(1024*1024)} MB", 413

    pdf_bytes = file.read()
    jd_text = (request.form.get("jd_text") or "").strip()

    resume_text = extract_text_from_pdf_bytes(pdf_bytes)
    model = init_model()
    r_json = llm_json(model, RESUME_PROMPT, resume_text)
    j_json = llm_json(model, JD_PROMPT, jd_text)

    r_skills = r_json.get("skills", [])
    r_meta = r_json.get("meta", {"has_contact": False, "has_sections": False})
    r_sen = seniority_bucket(r_json.get("seniority_indicators", []))

    jd_skills = j_json.get("required_skills", [])
    jd_priority = j_json.get("priority", {}) or {}
    jd_role = j_json.get("role_title", "Unknown Role")
    jd_sen = j_json.get("seniority_required", "mid")

    matches, gaps = fuzzy_coverage(r_skills, jd_skills)
    score10 = score_ats(matches, gaps, jd_priority, r_meta, r_sen, jd_sen)

    # Optional spreadsheet log
    company = (request.form.get("company") or "").strip()
    locations = request.form.get("locations") or ""
    if company:
        try:
            agent = get_sheet_agent()
            agent.append(company=company, position=jd_role, locations=locations, ats_score=score10)
        except Exception as e:
            print("Logging failed:", e)

    return f"Your ATS Score is: {score10}/10"

if __name__ == "__main__":
    app.config["MAX_CONTENT_LENGTH"] = MAX_PDF_BYTES + 64 * 1024
    port = int(os.getenv("PORT", "8000"))
    app.run(debug=True, host="127.0.0.1", port=port)

