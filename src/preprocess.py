import spacy
import re

#Load English lang model
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    '''
    Cleans the input text:
    - Lowercases text
    - Removes stopwords (and,the) and non-alpha tokens
    - Lemmatizes words (running --> run)
    '''

    doc = nlp(text.lower()) #lowercase + tokenize

    tokens = [token.text for token in doc if token.is_alpha]
    return " ".join(tokens)

def extract_relevant_sections(text):
    """
    Extract likely 'skills' or 'experience' sections from resumes or job descriptions.
    Returns a reduced version of the input for better matching.
    """

    # Normalize for processing
    text = text.lower()

    # Extract bullets or lines mentioning key patterns
    relevant_lines = []
    for line in text.splitlines():
        if any(kw in line for kw in ["skill", "experience", "responsible", "technologies", "tools", "proficient", "worked with", "used", "requirements"]):
            relevant_lines.append(line)

    # Also include bullet points that likely represent skills
    for line in text.splitlines():
        if re.match(r"^\s*[-•*]\s*", line):
            relevant_lines.append(line)

    return "\n".join(relevant_lines)
