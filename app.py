import streamlit as st
import json
import chromadb
from chromadb.utils import embedding_functions
import ollama

# Load JSON data from file
with open("kaggle_competitions.json", "r") as file:
    competition_data = json.load(file)

# Ensure the data is a list and extract first competition if needed
if isinstance(competition_data, list):
    competitions = competition_data  # Store all competitions if multiple exist
else:
    competitions = [competition_data]  # Convert single object to list

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="competition_db")
collection = client.get_or_create_collection("competitions")

# Get existing IDs to prevent duplicates
existing_ids = set(collection.get()['ids']) if collection.count() > 0 else set()

# Add data to ChromaDB only if it's not already present
for idx, comp in enumerate(competitions):
    comp_id = f"comp_{idx}"
    if comp_id not in existing_ids:
        comp_text = f"Title: {comp['title']}, Category: {comp['category']}, Prize: {comp['prize']}, Deadline: {comp['deadline']}, Link: {comp['link']}"
        collection.add(documents=[comp_text], ids=[comp_id])

def retrieve_info(query):
    results = collection.query(query_texts=[query], n_results=1)
    if results["documents"]:
        return results["documents"][0][0]
    return "No relevant information found."

def generate_response(query, chat_history):
    context = retrieve_info(query)
    prompt = f"You are an AI assistant. Use the following competition details to answer user queries.\n\nCompetition Details: {context}\n\nUser Query: {query}\n\nAnswer:"
    response = ollama.chat(model="gemma3:1b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Streamlit UI
st.title("Kaggle Competition Chatbot")
chat_history = st.session_state.get("chat_history", [])

user_input = st.text_input("Ask about the competition:")
if user_input:
    response = generate_response(user_input, chat_history)
    chat_history.append(f"User: {user_input}")
    chat_history.append(f"Bot: {response}")
    st.session_state["chat_history"] = chat_history

    st.write("Response:", response)

# Display chat history
st.subheader("Chat History")
for chat in chat_history:
    st.write(chat)