A Streamlit based Document Evaluater App.

Uses Locally downloaded Flan-T5 google/flan-t5-small model and Huggingface embedding model BAAI/bge-large-en-v1.5 . 

Can try with larger models. My system lagged for larger models due to storage issues. But larger models will be able to provide high accuracy. This is sample code for the process.

Document chunks and synopsis are passed to the prompt to reduce the token passed in the prompt.
Chunks evaluated on similarity search from Fiass db.
document is evaluated by the llm model directly.

Authentication is checked with Mysql and bycrpt for user login.

Document privacy for personal information are replaced using en_core_web_sm from nlp.


Dependencies to install:
streamlit
transformers
torch
spacy
langchain
langchain-community
langchain-huggingface
faiss-cpu           # or faiss-gpu depending on your hardware
PyMuPDF             # for PyMuPDFLoader
mysql-connector-python
bcrypt

Run the app with 
streamlit run app.py

Login is open but can check code with following credentials:
Username: test1
password: secure123
