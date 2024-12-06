import openai
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

print("Hello ds")

# OpenAI API anahtarınızı buraya girin
openai.api_key = "OPENAI_API_KEY"

def read_pdf(file_path):
    """PDF dosyasını metne çevir."""
    reader = PdfReader("/Users/erdemkok/Downloads/AhmetErdemKok-2.pdf")
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def scrape_website(url):
    """Şirket web sitesinden bilgi çek."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup.get_text()

def prepare_rag_system(cv_text, job_description, company_info):
    """RAG sistemini hazırla."""
    # Tüm verileri birleştir
    combined_text = f"CV:\n{cv_text}\n\nJob Description:\n{job_description}\n\nCompany Info:\n{company_info}"

    # Metni parçalara ayır
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(combined_text)

    # FAISS ile vektör depolama
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(docs, embeddings)

    # RAG zincirini oluştur
    retriever = vectorstore.as_retriever()
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    return qa_chain

def generate_interview_questions(qa_chain):
    """GPT-4 ile mülakat soruları oluştur."""
    prompt = (
        "Mülakat sırasında sorulabilecek 5 adet mantıklı soru oluştur. "
        "Her soru için gerekçelerini detaylı olarak belirt."
    )
    response = qa_chain.run(prompt)
    return response

def main():
    # Kullanıcıdan verileri al
    cv_path = input("CV dosya yolunu girin (PDF): ")
    job_description = input("İş ilanı detaylarını girin: ")
    company_url = input("Şirket web sitesi URL'sini girin: ")

    # CV'yi oku
    cv_text = read_pdf(cv_path)

    # Şirket bilgisini web scraping ile al
    company_info = scrape_website(company_url)

    # RAG sistemini hazırla
    qa_chain = prepare_rag_system(cv_text, job_description, company_info)

    # Mülakat sorularını oluştur
    questions = generate_interview_questions(qa_chain)
    print("\nMülakat Soruları ve Gerekçeleri:\n")
    print(questions)

if __name__ == "__main__":
    main()