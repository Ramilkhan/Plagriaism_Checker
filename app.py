import streamlit as st
import pdfplumber
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="PDF Similarity Checker", layout="wide")

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + " "
    return text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return set(text.split())

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

st.title("📘 PDF Similarity Checker (Plagiarism Detector)")
st.write("Upload 4 student PDF files to check similarity using **Jaccard Index**.")

uploaded_files = st.file_uploader(
    "Upload exactly 4 PDF files",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files and len(uploaded_files) == 4:
    st.success("✅ 4 files uploaded successfully!")
    
    texts = [clean_text(extract_text_from_pdf(f)) for f in uploaded_files]

    similarity_matrix = [[0 for _ in range(4)] for _ in range(4)]
    for i in range(4):
        similarity_matrix[i][i] = 1.0

    for (i, j) in combinations(range(4), 2):
        sim = jaccard_similarity(texts[i], texts[j])
        similarity_matrix[i][j] = sim
        similarity_matrix[j][i] = sim

    df = pd.DataFrame(
        similarity_matrix,
        index=[f"PDF {i+1}" for i in range(4)],
        columns=[f"PDF {i+1}" for i in range(4)]
    )

    st.subheader("🔍 Similarity Matrix")
    st.dataframe(df.style.background_gradient(cmap="YlGnBu"))

    st.subheader("📊 Heatmap Visualization")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

elif uploaded_files:
    st.warning("⚠️ Please upload **exactly 4 PDF files**.")
else:
    st.info("📥 Upload your PDF files to begin.")
