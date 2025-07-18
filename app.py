# app.py
import streamlit as st
import yake
from bake.keyword_extraction import extract_keywords as bake_extract
import scispacy
import spacy
from scispacy.umls_linking import UmlsEntityLinker

# Load SciSpacy model and UMLS linker
@st.cache_resource

def load_sci_model():
    nlp = spacy.load("en_core_sci_sm")
    linker = UmlsEntityLinker(resolve_abbreviations=True, name="umls")
    nlp.add_pipe("scispacy_linker", config={"linker": linker})
    return nlp, linker

nlp, linker = load_sci_model()

# Page setup
st.title("🔍 Hybrid Keyword Extractor (BAKE + YAKE + SciSpacy/UMLS)")

st.markdown("""
This app extracts precise biomedical and technical keywords using:
- **BAKE** (contextual BERT-based extraction)
- **YAKE!** (statistical keyword detection)
- **SciSpacy + UMLS** (NER and medical concept linking)

No external API needed. Runs locally. 
""")

# Input text
text = st.text_area("Paste your abstract or paragraph:", height=300)

if st.button("Extract Keywords") and text.strip():
    with st.spinner("Running keyword extraction..."):
        # --- 1. YAKE!
        yake_kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
        yake_keywords = [kw for kw, score in yake_kw_extractor.extract_keywords(text)]

        # --- 2. BAKE
        try:
            bake_keywords = bake_extract(text, top_k=10)
        except Exception:
            bake_keywords = []

        # --- 3. SciSpacy + UMLS
        doc = nlp(text)
        umls_keywords = list(set([ent.text for ent in doc.ents if len(ent.text.split()) <= 4]))

        # --- Combine and deduplicate
        all_keywords = set(yake_keywords + bake_keywords + umls_keywords)

        # --- Display results
        st.subheader("🔑 Top Extracted Keywords")
        st.write(f"**Total unique keywords:** {len(all_keywords)}")
        st.markdown("\n".join([f"- {kw}" for kw in all_keywords]))

        # Optional source breakdown
        st.subheader("📌 Source Breakdown")
        st.markdown("**YAKE!:** " + ", ".join(yake_keywords))
        st.markdown("**BAKE:** " + ", ".join(bake_keywords))
        st.markdown("**UMLS Concepts:** " + ", ".join(umls_keywords))

else:
    st.info("Enter text and click 'Extract Keywords' to begin.")
