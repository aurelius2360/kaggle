import streamlit as st
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("kaggle_faiss", embedding_model, allow_dangerous_deserialization=True)

# Streamlit UI
st.title("🤖 Kaggle Competition Finder")
st.write("Ask me about Kaggle competitions!")

query = st.text_input("Enter your query (e.g., 'Ongoing Data Science competitions with prize money')")
if st.button("Search"):
    results = vectorstore.similarity_search(query, k=5)
    
    if results:
        for result in results:
            comp = json.loads(result.page_content)
            st.write(f"### {comp['title']}")
            st.write(f"📅 **Deadline**: {comp['deadline']}")
            st.write(f"🏆 **Prize**: {comp['prize']}")
            st.write(f"🔗 [View Competition]({comp['link']})")
            st.write("---")
    else:
        st.write("❌ No relevant competitions found. Try another query!")
