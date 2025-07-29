# Smart Resume Matcher 💼🤖

A Streamlit app that matches resumes to job descriptions using semantic similarity powered by Sentence-BERT (MiniLM-L6-v2).

## Features
- Upload `.pdf`, `.docx`, or `.txt` resumes and job posts
- Clean UI with progress bar and score bands
- Powered by Sentence Transformers (MiniLM)
- Highlights strong, moderate, or weak matches

## To Run Locally
```bash
pip install -r requirements.txt
streamlit run src/app.py
