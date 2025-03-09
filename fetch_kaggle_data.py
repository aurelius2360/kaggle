import kaggle
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# Authenticate with Kaggle API
kaggle.api.authenticate()

# Fetch ongoing Kaggle competitions
competitions = kaggle.api.competitions_list()

# Extract relevant details
competition_data = []
for comp in competitions:
    comp_dict = {
        "title": comp.title,
        "category": comp.category,
        "prize": comp.reward,
        "deadline": str(comp.deadline),
        "link": f"https://www.kaggle.com/c/{comp.ref}"
    }
    competition_data.append(comp_dict)

# Save to JSON (Optional)
with open("kaggle_competitions.json", "w") as f:
    json.dump(competition_data, f, indent=4)

# Convert competitions to FAISS format
docs = [Document(page_content=json.dumps(comp), metadata={"source": "Kaggle"}) for comp in competition_data]

# Use Local Embeddings (Sentence Transformers)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save FAISS index
vectorstore.save_local("kaggle_faiss")

print("✅ Kaggle competition data saved & indexed successfully!")
