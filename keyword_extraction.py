# bake/keyword_extraction.py

import yake
import spacy
import scispacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util

import numpy as np
import re

# Load SciSpacy biomedical model and sentence transformer (compact)
nlp = spacy.load("en_core_sci_sm", disable=["parser", "tagger"])
model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight, cloud-friendly

# Configure YAKE
yake_kw_extractor = yake.KeywordExtractor(
    lan="en", n=1, dedupLim=0.9, top=20, features=None
)

def clean_text(text):
    return re.sub(r"\s+", " ", text.strip())

def extract_yake_keywords(text, top_k=15):
    keywords = yake_kw_extractor.extract_keywords(text)
    return [kw for kw, _ in sorted(keywords, key=lambda x: x[1])[:top_k]]

def extract_scispacy_entities(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if len(ent.text) > 2]

def compute_tfidf_scores(texts, top_k=15):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    tfidf = vectorizer.fit_transform(texts)
    feature_array = np.array(vectorizer.get_feature_names_out())

    results = []
    for row in tfidf:
        scores = row.toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        top_keywords = feature_array[top_indices]
        results.append(top_keywords.tolist())
    return results

def deduplicate_keywords(keywords, threshold=0.75):
    embeddings = model.encode(keywords, convert_to_tensor=True)
    retained = []
    for i, kw in enumerate(keywords):
        if all(util.cos_sim(embeddings[i], embeddings[j]) < threshold for j in range(i)):
            retained.append(kw)
    return retained

def extract_keywords(text, top_k=10):
    text = clean_text(text)
    if not text or len(text) < 50:
        return []

    # Extract candidate sets
    yake_keywords = extract_yake_keywords(text, top_k=20)
    scispacy_entities = extract_scispacy_entities(text)
    tfidf_keywords = compute_tfidf_scores([text], top_k=20)[0]

    # Combine + filter
    all_candidates = list(set(yake_keywords + scispacy_entities + tfidf_keywords))
    all_candidates = [kw.lower().strip() for kw in all_candidates if len(kw) > 2]

    # Deduplicate semantically
    final_keywords = deduplicate_keywords(all_candidates, threshold=0.75)

    # Return top_k
    return final_keywords[:top_k]
