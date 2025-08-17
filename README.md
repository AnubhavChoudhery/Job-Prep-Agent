# AI-Powered Job Application Pipeline

A comprehensive web application that helps job seekers find, analyze, and prepare for job opportunities using AI-powered resume matching and interview preparation.

##  Features

- Resume Analysis: Upload your PDF resume for AI-powered job matching
- Smart Job Search: Search LinkedIn jobs by role and location
- ATS Scoring: Get realistic compatibility scores (1-10) for each job
- Smart Ranking: Jobs automatically sorted by best match first
- Interview Preparation: Auto-generated interview prep documents for each company
- Export Ready: Download Excel reports and ZIP files with all documents
- Web Interface: Easy-to-use Gradio interface accessible from any browser

##  Setup Instructions

### Prerequisites

- Python 3.8 or higher
- IBM Watson X AI account and API credentials
- ScrapingDog API key for job scraping

### Quick Start

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   Copy `.env.example` to `.env` and fill in your API credentials:
   ```env
   # IBM Watson X AI Configuration
   WATSONX_API_KEY=your_watsonx_api_key_here
   WATSONX_URL=your_watsonx_url_here
   WATSONX_PROJECT_ID=your_project_id_here
   WATSONX_MODEL_ID=your_model_id_here

   # ScrapingDog API Configuration
   SCRAPINGDOG_API_KEY=your_scrapingdog_api_key_here
   ```

4. **Run the application:**
   ```bash
   python web_app.py
   ```

5. **Access the web interface:**
   - Local: `http://localhost:7860` (or any other port e.g. 8000)
   - Public link will be displayed in the terminal

##  How to Use

### Step 1: Upload Resume
- Upload your resume in PDF format
- Ensure it's well-formatted for best AI analysis

### Step 2: Specify Search Parameters
- Job Role: Enter the position you're looking for (e.g., "Software Engineer", "Data Scientist")
- Location: Specify location or "Remote" (e.g., "USA", "Canada", "Ireland")
- Number of Jobs: Choose how many jobs to analyze (steps of 5)

### Step 3: Start Analysis
- Click "Start Job Search & Analysis"
- Monitor progress through the status updates
- Process typically takes 2-5 minutes depending on number of jobs

### Step 4: Download Results
- Excel Report: Complete job listing with ATS scores, sorted by best matches
- Interview Documents: ZIP file containing Word documents with company-specific interview preparation

##  Output Files

### Excel Spreadsheet
Contains columns:
- Job title and company
- Location and job link
- Posted date
- ATS Score (X/10 format)
- Sorted by score (best matches first)

### Interview Documents (DOCX)
Each document includes:
- Interview process overview
- Common questions and topics
- Difficulty level and expectations
- Preparation tips and recommendations
- Key takeaways

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `WATSONX_API_KEY` | IBM Watson X AI API key | Yes |
| `WATSONX_URL` | Watson X AI service URL | Yes |
| `WATSONX_PROJECT_ID` | Your Watson X project ID | Yes |
| `WATSONX_MODEL_ID` | Model identifier to use | Yes |
| `SCRAPINGDOG_API_KEY` | ScrapingDog API for job scraping | Yes |

### Customization

You can modify the following in `web_app.py`:
- ATS_PROMPT: Customize how jobs are scored
- SUMMARIZE_JD_PROMPT: Adjust job description summarization
- Default values: Change default role, location, or job count
- UI theme: Modify the Gradio theme

##  Troubleshooting

### Common Issues

1. "No jobs found"
   - Try broader job role terms
   - Check if location is spelled correctly
   - Verify ScrapingDog API key is working

2. "Error processing jobs"
   - Check Watson X AI credentials
   - Ensure all environment variables are set
   - Verify internet connection

3. PDF upload issues
   - Ensure file is actually a PDF
   - Check file isn't corrupted
   - Try with a simpler PDF format

### API Rate Limits
- ScrapingDog: Check your plan limits
- Watson X AI: Monitor token usage
- Consider adding delays if hitting rate limits

##  Project Structure

```
job-application-pipeline/
├── web_app.py              # Main Gradio web interface
├── app.py                  # Original CLI pipeline
├── scrape.py              # Job scraping functionality
├── interview.py           # Interview document generation
├── location.py            # Location detection utilities
├── requirements.txt       # Python dependencies
├── setup.sh              # Setup script
├── .env                  # Environment variables (create this)
└── README.md             # This file
```

##  Contributing

Feel free to submit issues, feature requests, or pull requests to improve the application.
Mail us at anubhavchoudhery95@gmail.com and aadityashankar21@gmail.com

##  License

This project is open source under the MIT License. Please ensure you comply with the terms of service for all APIs used (WatsonX AI, ScrapingDog, etc.).

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify all environment variables are correctly set
3. Check API service status and quotas
4. Review error messages in the web interface

---

