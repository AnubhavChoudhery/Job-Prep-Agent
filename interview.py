from typing import List, Optional, NamedTuple
import os
import re
import time
import json
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import praw
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames


# -----------------------------
# Data Models
# -----------------------------
class RedditPost(NamedTuple):
    title: str
    content: str
    url: str
    score: int
    created_utc: float
    num_comments: int
    subreddit: str

class InterviewInsights(NamedTuple):
    common_questions: List[str]
    interview_process: str
    preparation_tips: List[str]
    focus_areas: List[str]  
    company_culture: str
    difficulty_rating: str
    recent_experiences: List[str]
    industry_specific_notes: str

class SearchConfig(NamedTuple):
    company_name: str
    position_name: str
    industry: str
    location: Optional[str] = None
    max_posts: int = 50
    days_back: int = 365


# -----------------------------
# Subreddit Mapping
# -----------------------------
INDUSTRY_SUBREDDITS = {
    'technology': [
        'cscareerquestions', 'ExperiencedDevs', 'ITCareerQuestions', 
        'datascience', 'MachineLearning', 'webdev', 'sysadmin'
    ],
    'finance': [
        'FinancialCareers', 'SecurityAnalysis', 'investing', 'accounting',
        'financialindependence', 'consulting', 'MBA'
    ],
    'healthcare': [
        'medicine', 'nursing', 'pharmacy', 'dentistry', 'medicalschool',
        'residency', 'healthcare'
    ],
    'consulting': ['consulting', 'MBA', 'strategy', 'businessanalysis'],
    'marketing': ['marketing', 'advertising', 'socialmedia', 'SEO', 'digitalmarketing'],
    'sales': ['sales', 'entrepreneur', 'smallbusiness', 'B2B'],
    'education': ['teaching', 'professors', 'education', 'academia', 'gradschool'],
    'legal': ['law', 'lawschool', 'lawyers', 'paralegal'],
    'design': ['graphic_design', 'userexperience', 'web_design', 'architecture'],
    'engineering': ['engineering', 'civilengineering', 'MechanicalEngineering', 
                    'ElectricalEngineering', 'ChemicalEngineering'],
    'media': ['journalism', 'writing', 'movies', 'television', 'advertising'],
    'retail': ['retail', 'smallbusiness', 'entrepreneur'],
    'government': ['publicservice', 'government', 'military', 'SecurityClearance'],
    'nonprofit': ['nonprofit', 'socialwork', 'volunteering'],
    'general': ['jobs', 'careerguidance', 'interviews', 'careerchange', 
                'findapath', 'getemployed', 'resumes']
}


# -----------------------------
# Reddit Auth (PRAW)
# -----------------------------
def create_reddit_client():
    return praw.Reddit(
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT", "UniversalInterviewBot/1.0")
    )


# -----------------------------
# Model Init
# -----------------------------
def init_model(WATSONX_URL, WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_MODEL_ID):
    creds = Credentials(url=WATSONX_URL, api_key=WATSONX_API_KEY)
    return ModelInference(
        model_id=WATSONX_MODEL_ID,
        credentials=creds,
        project_id=WATSONX_PROJECT_ID
    )


# -----------------------------
# Reddit Search Helpers
# -----------------------------
def get_industry_subreddits(industry: str) -> List[str]:
    industry_lower = industry.lower()
    if industry_lower in INDUSTRY_SUBREDDITS:
        return INDUSTRY_SUBREDDITS[industry_lower] + INDUSTRY_SUBREDDITS['general']
    for key, subreddits in INDUSTRY_SUBREDDITS.items():
        if key in industry_lower or industry_lower in key:
            return subreddits + INDUSTRY_SUBREDDITS['general']
    return INDUSTRY_SUBREDDITS['general']

def create_search_queries(config: SearchConfig) -> List[str]:
    company = config.company_name
    position = config.position_name
    location = config.location
    
    base_queries = [
        f"{company} interview",
        f"{company} hiring process",
        f"{company} {position} interview",
        f"interviewed at {company}",
        f"{company} application process",
        f"{position} interview experience",
        f"{position} interview questions"
    ]
    
    if location:
        base_queries.extend([
            f"{company} {location} interview",
            f"{position} {location} interview"
        ])
    
    return list(set(base_queries))


# -----------------------------
# Relevance Filter
# -----------------------------
def is_interview_related(text: str) -> bool:
    text_lower = text.lower()
    interview_indicators = [
        'interview', 'hiring', 'application process', 'recruiter', 'hr',
        'screening', 'assessment', 'offer', 'rejection', 'final round',
        'phone call', 'video call', 'zoom', 'teams', 'background check'
    ]
    return any(ind in text_lower for ind in interview_indicators)


# -----------------------------
# Fetch Posts (Parallel, PRAW)
# -----------------------------
def fetch_posts_parallel_praw(subreddits, queries, posts_per_query=3):
    reddit = create_reddit_client()

    def fetch_single(args):
        subreddit_name, query = args
        subreddit = reddit.subreddit(subreddit_name)
        results = []
        try:
            for submission in subreddit.search(query, sort="relevance", time_filter="year", limit=posts_per_query):
                if is_interview_related(submission.title + ' ' + (submission.selftext or "")):
                    results.append(RedditPost(
                        title=submission.title,
                        content=submission.selftext or "",
                        url=submission.url,
                        score=submission.score,
                        created_utc=submission.created_utc,
                        num_comments=submission.num_comments,
                        subreddit=subreddit_name
                    ))
        except Exception as e:
            print(f"Error searching r/{subreddit_name} for '{query}': {str(e)}")
        return results

    search_tasks = [(sr, q) for sr in subreddits for q in queries]
    all_posts = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for future in as_completed(executor.submit(fetch_single, task) for task in search_tasks):
            all_posts.extend(future.result())
    return all_posts


# -----------------------------
# Post Ranking & Dedup
# -----------------------------
def deduplicate_posts(posts: List[RedditPost]) -> List[RedditPost]:
    seen = set()
    unique_posts = []
    for post in posts:
        if post.url not in seen:
            seen.add(post.url)
            unique_posts.append(post)
    return unique_posts

def rank_posts_by_relevance(posts: List[RedditPost], config: SearchConfig) -> List[RedditPost]:
    def score_post(post):
        score = 0
        text = (post.title + " " + post.content).lower()
        if config.company_name.lower() in text: score += 10
        if config.position_name.lower() in text: score += 5
        if config.location and config.location.lower() in text: score += 3
        days_old = (time.time() - post.created_utc) / 86400
        if days_old < 30: score += 2
        score += min(post.score / 10, 5)
        score += min(post.num_comments / 5, 3)
        return score
    return sorted(posts, key=score_post, reverse=True)


# -----------------------------
# LLM Analysis Helper
# -----------------------------
def prepare_content_for_llm(posts: List[RedditPost], config: SearchConfig) -> str:
    top_posts = posts[:15]
    return "\n".join(
        f"Post {i+1} (Score: {p.score}, r/{p.subreddit}):\nTitle: {p.title}\nContent: {p.content[:800]}"
        for i, p in enumerate(top_posts)
    )

def llm_json(model, system_prompt: str, user_text: str, max_tokens=1000):
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
    except:
        return {}


# -----------------------------
# Main Insights Function
# -----------------------------
def generate_interview_insights(config: SearchConfig, watsonx_url, watsonx_api_key, watsonx_project_id, watsonx_model_id):
    model = init_model(watsonx_url, watsonx_api_key, watsonx_project_id, watsonx_model_id)
    subreddits = get_industry_subreddits(config.industry)
    queries = create_search_queries(config)

    print(f"Searching {len(subreddits)} subreddits with {len(queries)} queries...")
    posts = fetch_posts_parallel_praw(subreddits, queries, posts_per_query=3)
    posts = deduplicate_posts(posts)
    ranked = rank_posts_by_relevance(posts, config)
    top_posts = ranked[:config.max_posts]

    content_summary = prepare_content_for_llm(top_posts, config)
    system_prompt = f"You are an expert career counselor in {config.industry} interviews."
    resp = llm_json(model, system_prompt, content_summary, max_tokens=2000)

    return InterviewInsights(
        common_questions=resp.get("common_questions", []),
        interview_process=resp.get("interview_process", ""),
        preparation_tips=resp.get("preparation_tips", []),
        focus_areas=resp.get("focus_areas", []),
        company_culture=resp.get("company_culture", ""),
        difficulty_rating=resp.get("difficulty_rating", "Medium"),
        recent_experiences=resp.get("recent_experiences", []),
        industry_specific_notes=resp.get("industry_specific_notes", "")
    )


# -----------------------------
# Main Entrypoint
# -----------------------------
def main():
    load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_URL = os.getenv("WATSONX_URL")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID")

    config = SearchConfig(
        company_name="Microsoft",
        position_name="Software Engineer",
        industry="Technology",
        location="Seattle",
        max_posts=40
    )

    insights = generate_interview_insights(
        config,
        watsonx_url=WATSONX_URL,
        watsonx_api_key=WATSONX_API_KEY,
        watsonx_project_id=WATSONX_PROJECT_ID,
        watsonx_model_id=WATSONX_MODEL_ID
    )

    print("\n=== INTERVIEW INSIGHTS ===")
    print("Process:", insights.interview_process)
    print("Questions:", insights.common_questions)
    print("Focus Areas:", insights.focus_areas)
    print("Tips:", insights.preparation_tips)


if __name__ == "__main__":
    main()
