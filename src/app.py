import streamlit as st
from preprocess import clean_text, extract_relevant_sections
from matcher import get_semantic_match_score
import os
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

# Load external CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{file_name}' not found. Default styling will be used.")
local_css("src/styles.css")


def extract_file_text(uploaded_file):
    '''
    Reads uploaded file (.txt, .pdf, .docx) and returns raw text
    '''
    file_type = uploaded_file.name.lower()
    if file_type.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    
    elif file_type.endswith('.pdf'):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())
        text = extract_pdf_text("temp.pdf")
        os.remove("temp.pdf")
        return text
    
    elif file_type.endswith('.docx'):
        doc = Document(uploaded_file)
        return '\n'.join([para.text for para in doc.paragraphs])
    
    else:
        st.error("Unsupported file type. Please upload a .txt, .pdf, or .docx file.")
        return None
    
#page set up
st.set_page_config(page_title="Resume Matcher", layout="centered")
st.title("Resume vs Job Description Matcher")

#file upload
resume_file = st.file_uploader("Upload Resume (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="resume")
job_file = st.file_uploader("Upload Job Description (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="job")
if resume_file and job_file:
    #extract text from files
    resume_text = extract_file_text(resume_file)
    job_text = extract_file_text(job_file)
    
    if resume_text and job_text:
       # Focus only on relevant content to reduce dilution
        resume_focus = extract_relevant_sections(resume_text)
        job_focus = extract_relevant_sections(job_text)

        # Now clean the trimmed-down text
        cleaned_resume = clean_text(resume_focus)
        cleaned_job = clean_text(job_focus)
        
        #get match score
        semantic_score = get_semantic_match_score(cleaned_resume, cleaned_job)

        
        # Display results
        st.subheader("Results Summary")
        st.markdown(f"### ✅ Match Score: **{semantic_score * 100:.0f}%**")
        st.progress(semantic_score)
        if semantic_score >= 0.7:
            st.success("Strong match! 🔥")
        elif semantic_score >= 0.4:
            st.warning("Moderate match. Could be improved.")
        else:
            st.error("Low match. Try refining the resume with more relevant skills or terms.")

        st.markdown("### Resume Focused Content:")
        st.write(resume_focus)
        st.markdown("### Job Description Focused Content:")
        st.write(job_focus)
