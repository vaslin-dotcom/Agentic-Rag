# chunker.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_txt(txt, chunk_size=2000, overlap=200):  # ✅ better sizes
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]  # ✅ respects natural breaks
    )
    chunks = splitter.split_text(txt)
    return chunks

if __name__=='__main__':
    from loading_txt import extract_txt
    txt=extract_txt('MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf')
    chunks=chunk_txt(txt)
    print(len(chunks))


