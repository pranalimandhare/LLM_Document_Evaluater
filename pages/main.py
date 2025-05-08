import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
import spacy
import re
import tempfile
import torch

# Load NLP & Models
nlp = spacy.load("en_core_web_sm")
embed_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
model_path = "./Flan-T5"
tokenizer = AutoTokenizer.from_pretrained(model_path)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map={"": "cpu"})
llm = pipeline("text2text-generation", model=model_path, tokenizer=tokenizer, max_new_tokens=500)

# Helpers
def chunk_with_character_splitter(text, chunk_size=500, chunk_overlap=100):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def get_top_chunks(article_text, synopsis_text, top_k=4):
    article_chunks = chunk_with_character_splitter(article_text)
    db = FAISS.from_texts(article_chunks, embed_model)
    top_chunks_docs = db.similarity_search(synopsis_text, k=top_k)
    return [doc.page_content for doc in top_chunks_docs]

def get_llm_feedback(synopsis_text, top_chunks):
    llm_input = "\n\n".join(top_chunks)
    prompt = f"""
You are a document evaluator.

Evaluate the following synopsis based on the provided article chunks using the criteria below.
Write a detailed evaluation (at least 200 words), assign scores to each category, and give a final score for each header:

1. Content Coverage [out of 50]
2. Relevance [out of 50]
3. Clarity [out of 50]
4. Structure [out of 50]
5. Grammar & Language [out of 50]

Article Chunks:
{llm_input}

Synopsis:
{synopsis_text}
"""
    response = llm(prompt, temperature=0.2, max_new_tokens=500)[0]['generated_text']
    return response

def anonymize_text_general(text):
    doc = nlp(text)
    anonymized_text = text
    replacements = {}

    # Define personal-related entity labels (excluding CARDINAL, MONEY, DATE, etc.)
    personal_entity_labels = {"PERSON", "GPE", "LOC", "ORG"}

    for ent in doc.ents:
        if ent.label_ in personal_entity_labels:
            placeholder = f"<{ent.label_.upper()}>"
            if ent.text not in replacements:
                replacements[ent.text] = placeholder
                anonymized_text = anonymized_text.replace(ent.text, placeholder)

    # Anonymize common personal info patterns
    anonymized_text = re.sub(r'\b[\w\.-]+?@\w+?\.\w+?\b', '<EMAIL>', anonymized_text)
    anonymized_text = re.sub(r'\b\d{10,}\b', '<PHONE>', anonymized_text)
    anonymized_text = re.sub(r'http\S+', '<URL>', anonymized_text)

    return anonymized_text, replacements

# === Streamlit App ===
st.title("📄 Synopsis Evaluator with Anonymization")

pdf_file = st.file_uploader("Upload PDF Document", type=["pdf"])
txt_file = st.file_uploader("Upload Synopsis Text File", type=["txt"])
anonymize = st.checkbox("Anonymize Named Entities in Synopsis", value=True)

if st.button("Evaluate") and pdf_file and txt_file:
    # Save files temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(pdf_file.read())
        pdf_path = temp_pdf.name
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as temp_txt:
        temp_txt.write(txt_file.read())
        txt_path = temp_txt.name

    # Load article and synopsis
    article_pages = PyMuPDFLoader(pdf_path).load()
    article_text = " ".join([p.page_content for p in article_pages])

    synopsis_pages = TextLoader(txt_path).load()
    synopsis_text = " ".join([p.page_content for p in synopsis_pages])

    if anonymize:
        synopsis_text, replacements = anonymize_text_general(synopsis_text)
        for word, placeholder in replacements.items():
            synopsis_text = synopsis_text.replace(word, '')

    with st.spinner("Evaluating..."):
        top_chunks = get_top_chunks(article_text, synopsis_text)
        feedback = get_llm_feedback(synopsis_text, top_chunks)

    st.subheader("📝 Evaluation Feedback")
    st.text_area("Feedback", feedback, height=400)

    if anonymize:
        st.subheader("🔍 Replacements Made")
        for original, placeholder in replacements.items():
            st.markdown(f"**{original}** → `{placeholder}`")

