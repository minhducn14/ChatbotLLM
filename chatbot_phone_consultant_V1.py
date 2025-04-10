import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import chromadb

load_dotenv()

# === 1. Truy vấn dữ liệu từ SQL Server ===
def fetch_product_documents():
    server = "localhost"
    database = "ProductDB"
    username = "sa"
    password = "sapassword"

    conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"
    engine = create_engine(conn_str)

    query = text("""
        SELECT 
            p.name AS product_name,
            pv.name AS variant_name,
            pvd.price,
            pvd.status,
            a.name AS attribute_name,
            av.value AS attribute_value
        FROM products p
        JOIN products_variants pv ON p.id = pv.product_id
        JOIN product_variant_details pvd ON pv.id = pvd.product_variant_id
        JOIN attribute_values av ON av.product_variant_id = pv.id
        JOIN attributes a ON av.attribute_id = a.id
    """)

    with engine.connect() as conn:
        result = conn.execute(query).mappings().all()

    # Gom thuộc tính theo từng sản phẩm-phiên bản
    grouped = {}
    for row in result:
        key = f"{row['product_name']} - {row['variant_name']}"
        if key not in grouped:
            grouped[key] = {
                "price": row["price"],
                "status": row["status"],
                "attributes": []
            }
        grouped[key]["attributes"].append(f"{row['attribute_name']}: {row['attribute_value']}")

    # Tạo Document
    documents = []
    for key, value in grouped.items():
        content = f"{key} | Price: {value['price']} USD | Status: {value['status']}\n" + ", ".join(value["attributes"])
        documents.append(Document(page_content=content, metadata={}))

    return documents

# === 2. Khởi tạo Chroma vector store sử dụng embeddings miễn phí ===
def build_vector_store():
    docs = fetch_product_documents()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(docs)

    # Sử dụng HuggingFace embeddings (miễn phí) thay vì OpenAI
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    db = Chroma.from_documents(chunks, embedding_function, persist_directory="./chroma_phone_db")
    return db

# === 3. Prompt dành cho chatbot tư vấn điện thoại ===
prompt_template = PromptTemplate(
    template="""
You are a smartphone sales consultant chatbot.
Answer the customer's questions. When relevant questions come, use the provided documents. 
Please answer their specific question. If you are unsure, say "I'm not sure, please contact our customer support team."
Use friendly, courteous, and professional language like a sales advisor.
Keep your answers short and clear.

Context:
{context}

Question: {question}
Answer:
""",
    input_variables=["context", "question"]
)

# === 4. Hàm xử lý truy vấn người dùng ===
def run(input_text):
    chromadb.api.client.SharedSystemClient.clear_system_cache()
    persist_dir = "./chroma_phone_db"

    # Sử dụng HuggingFace embeddings khi load database
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_dir):
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
    else:
        db = build_vector_store()

    # Đã sử dụng Ollama (miễn phí)
    llm = OllamaLLM(model="llama3")

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt_template}
    )

    try:
        response = chain.invoke({"query": input_text})
        return response["result"] if "result" in response else response
    except Exception as e:
        return f"Error: {str(e)}"

# === Ví dụ chạy thử ===
if __name__ == "__main__":
    question = "Hiện tại tôi có bao nhiêu sản phẩm"
    print(run(question))