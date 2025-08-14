import praw
import time
from typing import Dict, List
from dataclasses import dataclass
from transformers import BartForConditionalGeneration, BartTokenizer
from dotenv import load_dotenv
from pathlib import Path
import torch
import praw
import os

@dataclass
class RedditPost:
    """Data class to store Reddit post information"""
    title: str
    content: str
    url: str
    subreddit: str
    score: int
    created_utc: float
    num_comments: int
    comments_text: str = ""

class JobInfoSummarizer:
    """BART-based summarizer for job-related Reddit content"""
    
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Summarize a single text using BART"""
        if len(text.strip()) < 50:  # Skip very short texts
            return text
        
        # Truncate if too long for BART
        max_input_length = 1024
        if len(text) > max_input_length * 4:  # Rough token estimation
            text = text[:max_input_length * 4]
        
        try:
            inputs = self.tokenizer.encode("summarize: " + text, 
                                         return_tensors="pt", 
                                         max_length=1024, 
                                         truncation=True)
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=2.0,
                    num_beams=4,
                    early_stopping=True
                )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary.strip()
        
        except Exception as e:
            print(f"Error summarizing text: {e}")
            return text[:200] + "..." if len(text) > 200 else text
    
    def create_comprehensive_summary(self, posts: List[RedditPost], 
                                   company_name: str, 
                                   position_name: str) -> Dict[str, str]:
        """Create categorized summaries from all posts"""
        
        # Categorize posts by content type
        categories = {
            'company_culture': [],
            'interview_experience': [],
            'salary_compensation': [],
            'work_experience': [],
            'career_advice': [],
            'general_discussion': []
        }
        
        # Keywords for categorization
        culture_keywords = ['culture', 'environment', 'workplace', 'team', 'management', 'work life balance']
        interview_keywords = ['interview', 'interviewed', 'hiring', 'recruiter', 'application', 'applied']
        salary_keywords = ['salary', 'pay', 'compensation', 'benefits', 'bonus', 'stock', 'equity']
        experience_keywords = ['experience', 'working at', 'employee', 'job', 'role', 'position']
        advice_keywords = ['advice', 'tips', 'should i', 'how to', 'recommend', 'suggestion']
        
        for post in posts:
            full_text = f"{post.title} {post.content} {post.comments_text}".lower()
            
            if any(keyword in full_text for keyword in interview_keywords):
                categories['interview_experience'].append(post)
            elif any(keyword in full_text for keyword in salary_keywords):
                categories['salary_compensation'].append(post)
            elif any(keyword in full_text for keyword in culture_keywords):
                categories['company_culture'].append(post)
            elif any(keyword in full_text for keyword in advice_keywords):
                categories['career_advice'].append(post)
            elif any(keyword in full_text for keyword in experience_keywords):
                categories['work_experience'].append(post)
            else:
                categories['general_discussion'].append(post)
        
        # Generate summaries for each category
        summaries = {}
        for category, category_posts in categories.items():
            if not category_posts:
                continue
            
            # Combine all relevant text for this category
            combined_text = ""
            for post in category_posts:
                post_text = f"Post: {post.title}\nContent: {post.content}\nComments: {post.comments_text[:500]}\n\n"
                combined_text += post_text
            
            if len(combined_text.strip()) > 100:
                category_summary = self.summarize_text(
                    combined_text, 
                    max_length=200, 
                    min_length=75
                )
                summaries[category] = {
                    'summary': category_summary,
                    'post_count': len(category_posts),
                    'total_score': sum(post.score for post in category_posts)
                }
        
        return summaries

def scrape_reddit_for_job_info(
    reddit: praw.Reddit,
    job_description: str,
    company_name: str,
    position_name: str,
    max_posts_per_search: int = 50,
    get_comments: bool = True,
    max_comments_per_post: int = 10
) -> List[RedditPost]:
    """
    Scrape Reddit for job-related posts and extract all relevant information.
    
    Args:
        reddit: Authenticated PRAW Reddit instance
        job_description: Description of the job/role
        company_name: Name of the company
        position_name: Position/job title
        max_posts_per_search: Maximum posts to fetch per search
        get_comments: Whether to fetch comments from posts
        max_comments_per_post: Maximum comments to fetch per post
    
    Returns:
        List of RedditPost objects with all extracted information
    """
    
    all_posts = []
    seen_urls = set()
    
    # Create comprehensive search terms
    search_terms = [
        f'"{company_name}"',
        f'"{position_name}"',
        f'{company_name} {position_name}',
        f'{company_name} interview',
        f'{company_name} work',
        f'{company_name} employee',
        f'{company_name} culture',
        f'{company_name} salary',
        f'{position_name} at {company_name}',
        f'working at {company_name}'
    ]
    
    # Extract key terms from job description
    job_keywords = extract_key_terms(job_description)
    for keyword in job_keywords[:5]:
        search_terms.append(f'{keyword} {company_name}')
    
    # Relevant subreddits for comprehensive coverage
    target_subreddits = [
        'all',  # Site-wide search
        'jobs',
        'cscareerquestions',
        'ITCareerQuestions', 
        'careerguidance',
        'ExperiencedDevs',
        'consulting',
        'sysadmin',
        'programming',
        'webdev',
        'datascience',
        'MachineLearning',
        'sales',
        'marketing',
        'finance',
        'business',
        'entrepreneur'
    ]
    
    print(f"Starting comprehensive Reddit scrape for {position_name} at {company_name}...")
    
    # Search across multiple subreddits and terms
    for subreddit_name in target_subreddits:
        print(f"Searching r/{subreddit_name}...")
        
        try:
            subreddit = reddit.subreddit(subreddit_name)
            
            # Search with different terms
            for search_term in search_terms[:3]:  # Limit to avoid rate limits
                try:
                    search_results = subreddit.search(
                        search_term,
                        sort='relevance',
                        time_filter='year',  # Look at past year
                        limit=max_posts_per_search
                    )
                    
                    for post in search_results:
                        if post.url in seen_urls:
                            continue
                        
                        seen_urls.add(post.url)
                        
                        # Extract post content
                        post_content = extract_post_content(post)
                        
                        # Get comments if requested
                        comments_text = ""
                        if get_comments:
                            comments_text = extract_comments(post, max_comments_per_post)
                        
                        reddit_post = RedditPost(
                            title=post.title,
                            content=post_content,
                            url=post.url,
                            subreddit=post.subreddit.display_name,
                            score=post.score,
                            created_utc=post.created_utc,
                            num_comments=post.num_comments,
                            comments_text=comments_text
                        )
                        
                        all_posts.append(reddit_post)
                    
                    time.sleep(1.5)  # Rate limiting between searches
                    
                except Exception as e:
                    print(f"Error searching '{search_term}' in r/{subreddit_name}: {e}")
                    continue
            
            time.sleep(2)  # Longer delay between subreddits
            
        except Exception as e:
            print(f"Error accessing r/{subreddit_name}: {e}")
            continue
    
    print(f"Scraping completed. Found {len(all_posts)} unique posts.")
    return all_posts

def extract_post_content(post) -> str:
    """Extract all relevant text content from a Reddit post"""
    content_parts = []
    
    # Add post title
    if hasattr(post, 'title') and post.title:
        content_parts.append(f"Title: {post.title}")
    
    # Add post body text
    if hasattr(post, 'selftext') and post.selftext:
        content_parts.append(f"Content: {post.selftext}")
    
    # If it's a link post, note the URL
    if hasattr(post, 'url') and post.url and not post.is_self:
        content_parts.append(f"Link: {post.url}")
    
    return "\n\n".join(content_parts)

def extract_comments(post, max_comments: int = 10) -> str:
    """Extract top comments from a Reddit post"""
    try:
        post.comments.replace_more(limit=0)  # Remove "load more comments"
        
        comments_text = []
        comment_count = 0
        
        for comment in post.comments[:max_comments]:
            if hasattr(comment, 'body') and comment.body != '[deleted]':
                comments_text.append(f"Comment ({comment.score} points): {comment.body}")
                comment_count += 1
                if comment_count >= max_comments:
                    break
        
        return "\n\n".join(comments_text)
    
    except Exception as e:
        print(f"Error extracting comments: {e}")
        return ""

def extract_key_terms(job_description: str) -> List[str]:
    """Extract key terms from job description for better search"""
    # Common tech terms and skills
    tech_terms = [
        'python', 'java', 'javascript', 'react', 'node.js', 'sql', 'aws', 'docker',
        'kubernetes', 'machine learning', 'data science', 'tensorflow', 'pytorch',
        'django', 'flask', 'spring', 'angular', 'vue', 'mongodb', 'postgresql',
        'devops', 'agile', 'scrum', 'api', 'microservices', 'cloud', 'linux'
    ]
    
    job_lower = job_description.lower()
    found_terms = []
    
    for term in tech_terms:
        if term in job_lower:
            found_terms.append(term)
    
    return found_terms[:10]

def generate_job_insights(
    reddit: praw.Reddit,
    job_description: str,
    company_name: str,
    position_name: str,
    max_posts: int = 100
) -> Dict[str, any]:
    """
    Main function to scrape Reddit and generate AI-powered insights
    """
    
    # Step 1: Scrape all relevant posts
    print("Phase 1: Scraping Reddit posts...")
    posts = scrape_reddit_for_job_info(
        reddit, job_description, company_name, position_name, max_posts
    )
    
    if not posts:
        return {"error": "No relevant posts found"}
    
    # Step 2: Initialize BART summarizer
    print("Phase 2: Initializing BART summarizer...")
    summarizer = JobInfoSummarizer()
    
    # Step 3: Generate comprehensive summaries
    print("Phase 3: Generating AI summaries...")
    summaries = summarizer.create_comprehensive_summary(posts, company_name, position_name)
    
    # Step 4: Compile final insights
    insights = {
        'company': company_name,
        'position': position_name,
        'total_posts_analyzed': len(posts),
        'data_sources': list(set(post.subreddit for post in posts)),
        'summaries': summaries,
        'raw_posts_sample': [
            {
                'title': post.title,
                'subreddit': post.subreddit,
                'score': post.score,
                'url': post.url
            } for post in sorted(posts, key=lambda x: x.score, reverse=True)[:5]
        ]
    }
    
    return insights

# Initialize Reddit API
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
reddit = praw.Reddit(
    client_id=os.getenv('CLIENT_ID'),
    client_secret=os.getenv('CLIENT_SECRET'),
    user_agent=os.getenv('REDDIT_USER_AGENT')
)

# Generate insights
job_desc = "Looking for Senior Data Scientist with Python, ML, and AWS experience..."
insights = generate_job_insights(reddit, job_desc, "Netflix", "Senior Data Scientist")

# Print results
print(f"Analyzed {insights['total_posts_analyzed']} posts from {len(insights['data_sources'])} subreddits")
for category, data in insights['summaries'].items():
    print(f"\n{category.upper()}:")
    print(f"Summary: {data['summary']}")
    print(f"Based on {data['post_count']} posts")
