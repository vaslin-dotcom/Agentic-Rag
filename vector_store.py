

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from tqdm import tqdm

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={
        "batch_size": 64,
        "normalize_embeddings": True
    }
)

chroma_path = 'chroma_db'

def build_vector_store(chunks):
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

    print(f"Total chunks to embed: {len(documents)}")

    # ✅ Add in batches with progress bar
    batch_size = 64
    vector_store = None

    for i in tqdm(range(0, len(documents), batch_size), 
                  desc="Building vector store",
                  unit="batch"):

        batch = documents[i:i + batch_size]

        if vector_store is None:
            # First batch — create the store
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=chroma_path
            )
        else:
            # Subsequent batches — add to existing store
            vector_store.add_documents(batch)

    print("Vector store created ✅")
    

def load_vector_store():
    print(f"Loading vector store from {chroma_path}")
    return Chroma(
        persist_directory=chroma_path,
        embedding_function=embeddings
    )

if __name__ == "__main__":
    from loading_txt import extract_txt
    from chunker import chunk_txt

    txt = extract_txt("MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf")
    print(f"Extracted {len(txt)} characters")  # ✅ sanity check

    chunks = chunk_txt(txt)
    print(f"Created {len(chunks)} chunks")     # ✅ sanity check

    build_vector_store(chunks)