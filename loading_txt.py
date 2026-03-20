# loading_txt.py
from pypdf import PdfReader

def extract_txt(file):
    reader = PdfReader(file)
    txt = ''
    for page in reader.pages[84:]:
        extracted_txt = page.extract_text()
        if extracted_txt:  # ✅ fixed bug
            txt += extracted_txt
    return txt

if __name__=='__main__':
    file='MahabharataOfVyasa-EnglishTranslationByKMGanguli.pdf'
    txt=extract_txt(file)
    print(txt[:300])