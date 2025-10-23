import pdfplumber
import re
import pandas as pd
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1️⃣ Extract text from PDF
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    """
    Extracts text content from a PDF file.
    Works with both file paths and Streamlit uploaded files.
    """
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text


# -----------------------------
# 2️⃣ Clean text
# -----------------------------
def clean_text(text):
    """
    Converts text to lowercase and removes non-alphabetic characters.
    Returns both cleaned string and token set for flexibility.
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return " ".join(words), set(words)


# -----------------------------
# 3️⃣ Jaccard Similarity
# -----------------------------
def jaccard_similarity(set1, set2):
    """
    Computes Jaccard similarity between two sets of words.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0


# -----------------------------
# 4️⃣ Cosine Similarity (TF-IDF)
# -----------------------------
def cosine_similarity_score(texts):
    """
    Computes cosine similarity matrix using TF-IDF representation.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    cosine_sim_matrix = cosine_similarity(tfidf_matrix)
    return cosine_sim_matrix


# -----------------------------
# 5️⃣ Compute Both Similarities
# -----------------------------
def compute_similarities(uploaded_files):
    """
    Takes a list of uploaded PDF files, extracts and cleans text,
    and returns both Jaccard and Cosine similarity DataFrames.
    """
    # Step 1: Extract and clean text
    cleaned_texts = []
    token_sets = []
    for file in uploaded_files:
        raw_text = extract_text_from_pdf(file)
        cleaned_str, token_set = clean_text(raw_text)
        cleaned_texts.append(cleaned_str)
        token_sets.append(token_set)

    # Step 2: Compute Jaccard matrix
    n = len(uploaded_files)
    jaccard_matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        jaccard_matrix[i][i] = 1.0
    for (i, j) in combinations(range(n), 2):
        sim = jaccard_similarity(token_sets[i], token_sets[j])
        jaccard_matrix[i][j] = sim
        jaccard_matrix[j][i] = sim

    # Step 3: Compute Cosine matrix
    cosine_matrix = cosine_similarity_score(cleaned_texts)

    # Step 4: Convert to DataFrames
    labels = [f"PDF {i+1}" for i in range(n)]
    jaccard_df = pd.DataFrame(jaccard_matrix, index=labels, columns=labels)
    cosine_df = pd.DataFrame(cosine_matrix, index=labels, columns=labels)

    return jaccard_df, cosine_df
