
from langchain_community.tools import DuckDuckGoSearchResults
from vector_store import *
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import pickle

with open('db_with_keywords.pkl','rb') as f:
    documents=pickle.load(f)

vector_store=load_vector_store()
vector_retriever=vector_store.as_retriever(search_kwargs={'k':4})
keyword_retriever=BM25Retriever.from_documents(documents,k=4)
ensemble_retriever=EnsembleRetriever(
    retrievers=[vector_retriever,keyword_retriever],
    weights=[0.70,0.30]
)


def hybrid_search(query:str)->list:

    results=ensemble_retriever.invoke(query)
    return [doc.page_content for doc in results]    


def web_search(query: str) -> list:
 
    search_tool = DuckDuckGoSearchResults(
        num_results=4,          # 👈 returns 4 results
        output_format='list'    # 👈 returns as list instead of single string
    )
    results = search_tool.invoke(query)
    # each result has snippet, title, link — extract just the snippet
    return [r['snippet'] for r in results if 'snippet' in r]

if __name__ == '__main__':
    # test hybrid
    results = hybrid_search("What was the role of Krishna in the Kurukshetra war?")
    for i, chunk in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk[:300])    # first 300 chars of each chunk
    print('==============================')
    # test web
    result = web_search("When was the Mahabharata written historically?")
    print(result)