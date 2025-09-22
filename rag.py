# =====================================
# STEP 0: Import Libraries & Setup
# =====================================
import os
import pandas as pd
from dotenv import load_dotenv
import time

# LangChain imports (new versions)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from groq import Groq

# Load API keys from .env file
load_dotenv()

# =====================================
# STEP 1: Load & Clean Dataset
# =====================================
df = pd.read_json(
    "data/flipkart_fashion_products_dataset.json",
    dtype=str,
    encoding="utf-8"
)

df = df[['actual_price', 'average_rating', 'brand', 'category', 'url']]
print("✅ Dataset loaded with shape:", df.shape)

# Reduce to only 20 rows for testing
df = df.head(50)
print("✅ Reduced to test set:", df.shape)

# =====================================
# STEP 2: Convert rows → Documents
# =====================================
documents = [
    Document(
        page_content=f"Brand: {row['brand']}, Price: {row['actual_price']}, "
                     f"Rating: {row['average_rating']}, Category: {row['category']}, "
                     f"URL: {row['url']}"
    )
    for _, row in df.iterrows()
]
print("✅ Documents created:", len(documents))
print("Example Document:", documents[0])

# =====================================
# STEP 2.1: Chunk Documents
# =====================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # larger chunks
    chunk_overlap=20  # minimal overlap
)
chunked_documents = text_splitter.split_documents(documents)
print("✅ Documents after chunking:", len(chunked_documents))

# =====================================
# STEP 3: Create Embeddings with Google API
# =====================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =====================================
# STEP 4: Store in FAISS Vector Database with Batching
# =====================================
def batch_process_documents(documents, batch_size=5):
    """Process documents in smaller batches with retry logic"""
    vectorstore = None
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        retries = 3
        
        while retries > 0:
            try:
                partial_vectorstore = FAISS.from_documents(batch, embeddings)
                if vectorstore is None:
                    vectorstore = partial_vectorstore
                else:
                    vectorstore.merge_from(partial_vectorstore)
                print(f"✅ Processed batch {i//batch_size + 1}")
                break
            except Exception as e:
                retries -= 1
                if retries == 0:
                    print(f"❌ Failed to process batch {i//batch_size + 1}: {e}")
                    continue
                print(f"⚠️ Retry {3-retries}/3 for batch {i//batch_size + 1}")
                time.sleep(2)  # Wait before retry
                
    return vectorstore

# Process in batches
vectorstore = batch_process_documents(chunked_documents)
if vectorstore:
    vectorstore.save_local("faiss_index")
    print("✅ FAISS index created with", vectorstore.index.ntotal, "embeddings")
else:
    print("❌ Failed to create vector store")
    exit(1)

# =====================================
# STEP 5: Retrieve Relevant Documents
# =====================================
query = "What is the price of Yorker shirts?"
retriever = vectorstore.as_retriever()
docs = retriever.get_relevant_documents(query)

print("✅ Retrieved", len(docs), "documents for query")
for d in docs[:2]:
    print("-", d.page_content)

# =====================================
# STEP 6: Ask Groq LLM with Retrieved Context
# =====================================
try:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    context = " ".join([doc.page_content for doc in docs[:3]])
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on this context, {query}\n\nContext: {context}"}
        ],
        model="llama-3.3-70b-versatile",
    )

    print("\n🤖 Final Answer from Groq:")
    print(chat_completion.choices[0].message.content)
except Exception as e:
    print("❌ Error getting response from Groq:", e)