import os
from preprocess import clean_text, extract_relevant_sections
from sentence_transformers import SentenceTransformer, util

# Load the model once globally
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_semantic_match_score(resume_text, job_text):
    """
    Computes the cosine similarity between the semantic embeddings
    of the resume and job description using Sentence-BERT.
    Returns a float score between 0 and 1 (rounded to 2 decimals).
    """
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_emb = model.encode(job_text, convert_to_tensor=True)

    score = util.cos_sim(resume_emb, job_emb)
    return round(score.item(), 2)



def read_and_clean(file_path):
    '''
    Reads a .txt file and returns cleaned text.
    '''

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    return clean_text(raw_text)

'''
def get_match_score_debug(resume_text, job_text): #attempt at debugging
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    
    vectorizer = TfidfVectorizer(
        lowercase = False,
        token_pattern = r"(?u)\b\w+\b",
        stop_words = None
    )

    vectors = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(vectors[0], vectors[1])
    
    #debug part

    feature_names = vectorizer.get_feature_names_out()
    resume_vector = vectors[0].toarray()[0]
    job_vector = vectors[1].toarray()[0]

    print("debugging info")
    print(f"Total features: {len(feature_names)}")
    print(f"Resume non-zero features: {np.count_nonzero(resume_vector)}")
    print(f"Job non-zero features: {np.count_nonzero(job_vector)}")

    #shows overlapping terms

    overlapping_terms = []
    for i, term in enumerate(feature_names):
        if resume_vector[i] > 0 and job_vector[i] > 0:
            overlapping_terms.append((term, resume_vector[i], job_vector[i]))

    print(f"Overlapping terms: {len(overlapping_terms)}")
    for term, resume_tfidf, job_tfidf in overlapping_terms:
        print(f" {term}: resume={resume_tfidf: .4f}, job={job_tfidf: .4f}")
    
    return {
    "debug_score": round(score[0][0], 2),
    "overlapping_terms": overlapping_terms,
    "feature_names": feature_names,
}
'''
# Test with your example data
if __name__ == "__main__":
    from preprocess import clean_text, extract_relevant_sections

    resume_path = "data/resumes/resume_1.txt"
    job_path = "data/jobs/job_1.txt"

    try:
        # Load raw text
        with open(resume_path, "r", encoding="utf-8") as f:
            resume_text = f.read()
        with open(job_path, "r", encoding="utf-8") as f:
            job_text = f.read()
    except FileNotFoundError as e:
        print(e)
        exit()

    # Extract focused sections and clean
    resume_focus = extract_relevant_sections(resume_text)
    job_focus = extract_relevant_sections(job_text)

    cleaned_resume = clean_text(resume_focus)
    cleaned_job = clean_text(job_focus)

    # Get semantic match score

    print("\nSemantic Match Score:")
    semantic_score = get_semantic_match_score(cleaned_resume, cleaned_job)
    print(f"Score: {semantic_score}")
