import requests
import os
import pandas as pd

API_KEY = os.getenv("SCRAPINGDOG_API_KEY")

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

if __name__ == "__main__":
    role = "Software Engineering Intern"
    location = "Ireland"
    
    jobs = get_linkedin_jobs(role, location)
    
    if jobs:
        df = pd.DataFrame(jobs)
        df.to_excel("linkedin_jobs_test.xlsx", index=False)
        
        # Show the jobs found
        print("\nJobs found:")
        for i, job in enumerate(jobs, 1):
            print(f"{i}. {job['title']} at {job['company']}")
    else:
        print("No jobs found")