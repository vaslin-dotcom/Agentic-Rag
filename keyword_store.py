import pickle
from langchain_core.documents import Document
from loading_txt import extract_txt
from chunker import chunk_txt

# Step 1 - Extract text from PDF
print("Extracting text from PDF...")
txt = extract_txt("MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf")
print(f"Extracted {len(txt)} characters")

# Step 2 - Chunk the text
print("Chunking text...")
chunks = chunk_txt(txt)
print(f"Created {len(chunks)} chunks")

# Step 3 - Convert to Document objects
print("Converting to Document objects...")
documents = [
    Document(
        page_content=chunk,
        metadata={
            "chunk_id": i,
            "source": "Mahabharata",
            "chunk_size": len(chunk)
        }
    )
    for i, chunk in enumerate(chunks)
]

# Step 4 - Save to pickle
print("Saving to documents.pkl...")
with open("db_with_keywords.pkl", "wb") as f:
    pickle.dump(documents, f)

print(f"Saved {len(documents)} documents to documents.pkl ")