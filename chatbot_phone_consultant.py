import os
import re
import json
import hashlib
import asyncio
import spacy
import logging
import logging.handlers
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List, Dict
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential

# --- Cấu hình logging với encoding UTF-8 ---
error_handler = logging.handlers.RotatingFileHandler(
    "error.log", maxBytes=1048576, backupCount=5, encoding="utf-8"
)
error_handler.setLevel(logging.ERROR)
error_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
error_handler.setFormatter(error_formatter)
logging.basicConfig(level=logging.INFO)  # Đổi level về INFO để in log debug
logger = logging.getLogger(__name__)
logger.addHandler(error_handler)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- Khởi tạo NLP ---
nlp = spacy.load("vi_core_news_lg")

# --- Cache ---
sql_cache = TTLCache(maxsize=10, ttl=3600)             # Cache truy vấn SQL (TTL 1 giờ)
product_list_cache = TTLCache(maxsize=1, ttl=3600)       # Cache danh sách sản phẩm (TTL 1 giờ)

# --- 1. Hàm truy vấn dữ liệu từ SQL Server (bất đồng bộ) ---
async def fetch_product_documents_async(page=1, page_size=100):
    cache_key = f"products_page_{page}_size_{page_size}"
    if cache_key in sql_cache:
        logger.info("Lấy dữ liệu từ cache cho trang %d", page)
        return sql_cache[cache_key]
    
    try:
        # Lấy thông tin kết nối từ biến môi trường hoặc sử dụng giá trị mặc định
        server = os.getenv("DB_SERVER", "localhost")
        database = os.getenv("DB_DATABASE", "ProductDB")
        username = os.getenv("DB_USERNAME", "sa")
        password = os.getenv("DB_PASSWORD", "sapassword")
        driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        driver_encoded = driver.replace(" ", "+")
        # Sử dụng scheme mssql+pyodbc
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver_encoded}"
        
        engine = create_engine(conn_str)
        offset = (page - 1) * page_size
        query = """
            SELECT 
                p.name AS product_name,
                pv.name AS variant_name,
                pv.name + ' - ' + c.name + ' - ' + m.name AS pvd_name,
                pvd.price,
                pvd.status,
                c.name AS color_name,
                m.name AS memory_name,
                STRING_AGG(a.name + ': ' + av.value, ', ') AS attributes
            FROM products p
            JOIN products_variants pv ON p.id = pv.product_id
            JOIN product_variant_details pvd ON pv.id = pvd.product_variant_id
            JOIN attribute_values av ON av.product_variant_id = pv.id
            JOIN attributes a ON av.attribute_id = a.id
            JOIN colors c ON c.id = pvd.color_id
            JOIN memories m ON pvd.memory_id = m.id
            GROUP BY p.name, pv.name, pv.name + ' - ' + c.name + ' - ' + m.name, pvd.price, pvd.status, c.name, m.name
            ORDER BY p.name, pv.name, pv.name + ' - ' + c.name + ' - ' + m.name, pvd.price, pvd.status, c.name, m.name
            OFFSET :offset ROWS FETCH NEXT :page_size ROWS ONLY
        """
        # Đóng gói hàm đồng bộ để chạy trong thread
        def sync_query():
            with engine.connect() as conn:
                result = conn.execute(text(query), {"offset": offset, "page_size": page_size}).mappings().all()
                return result
        result = await asyncio.to_thread(sync_query)
        
        documents = []
        logger.info("Trang %d: Số lượng sản phẩm lấy được: %d", page, len(result))
        for row in result:
            product_name = row["product_name"]
            variant_name = row["variant_name"]
            pvd_name = row["pvd_name"]
            content = (
                f"Tên sản phẩm: {product_name}\n"
                f"Tên biến thể: {variant_name}\n"
                f"Thông tin sản phẩm: {pvd_name}\n"
                f"Giá: {row['price']}\n"
                f"Tình trạng: {row['status']}\n"
                f"Thuộc tính: {row['attributes']}\n"
            )
            documents.append(Document(
                page_content=content, 
                metadata={"product_name": product_name, "variant_name": variant_name, "pvd_name": pvd_name}
            ))
        
        sql_cache[cache_key] = documents
        return documents
    except Exception as e:
        logger.error("Lỗi khi lấy dữ liệu sản phẩm: %s", e)
        return []

# --- 2. Xây dựng vector store (bất đồng bộ) ---
def compute_data_hash(documents):
    data_str = json.dumps([doc.page_content for doc in documents], sort_keys=True)
    return hashlib.sha256(data_str.encode('utf-8')).hexdigest()

async def build_vector_store_async():
    persist_dir = "./chroma_phone_db"
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    page = 1
    all_docs = []
    while True:
        docs = await fetch_product_documents_async(page=page, page_size=100)
        if not docs:
            break
        logger.info("Đã tải được %d sản phẩm từ trang %d", len(docs), page)
        all_docs.extend(docs)
        page += 1
    
    if not all_docs:
        logger.warning("Không tìm thấy tài liệu từ cơ sở dữ liệu.")
        return None

    new_data_hash = compute_data_hash(all_docs)
    hash_file = os.path.join(persist_dir, "data_hash.txt")
    
    if os.path.exists(persist_dir) and os.path.exists(hash_file):
        with open(hash_file, 'r', encoding="utf-8") as f:
            old_data_hash = f.read().strip()
        if old_data_hash == new_data_hash:
            logger.info("Dữ liệu không thay đổi, tái sử dụng vector store hiện có.")
            return Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
        
        # Nếu dữ liệu thay đổi, cập nhật vector store
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
        existing_docs = db.get(include=["documents", "metadatas"])
        existing_doc_ids = existing_docs["ids"]
        existing_doc_map = {meta["pvd_name"]: doc_id for doc_id, meta in zip(existing_doc_ids, existing_docs["metadatas"])}
        
        new_doc_map = {doc.metadata["pvd_name"]: doc for doc in all_docs}
        docs_to_add = []
        ids_to_delete = []
        for pvd_name, doc in new_doc_map.items():
            if pvd_name not in existing_doc_map:
                docs_to_add.append(doc)
        for pvd_name, doc_id in existing_doc_map.items():
            if pvd_name not in new_doc_map:
                ids_to_delete.append(doc_id)
        
        if ids_to_delete:
            db.delete(ids=ids_to_delete)
            logger.info("Đã xóa %d tài liệu khỏi vector store.", len(ids_to_delete))
        
        if docs_to_add:
            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = splitter.split_documents(docs_to_add)
            db.add_documents(chunks)
            logger.info("Đã thêm %d tài liệu mới vào vector store.", len(docs_to_add))
        
        with open(hash_file, 'w', encoding="utf-8") as f:
            f.write(new_data_hash)
        return db
    
    logger.info("Xây dựng vector store mới...")
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=persist_dir)
    os.makedirs(persist_dir, exist_ok=True)
    with open(hash_file, 'w', encoding="utf-8") as f:
        f.write(new_data_hash)
    return db

# --- 3. Định nghĩa trạng thái ---
class ConversationState(TypedDict):
    messages: List[Dict[str, str]]  # Lịch sử hội thoại

class ProductState(TypedDict):
    full_product_list: str
    product_list_with_index: List[tuple]

class ChatState(TypedDict):
    conversation: ConversationState
    products: ProductState
    question: str
    answer: str

# --- 4. Prompt chatbot ---
prompt_template = PromptTemplate(
    template="""
Bạn là một chuyên viên tư vấn bán điện thoại thông minh chuyên nghiệp.
**CHỈ SỬ DỤNG DỮ LIỆU ĐƯỢC CUNG CẤP TRONG CONTEXT** mà không bổ sung bất kỳ kiến thức nào từ bên ngoài.
Hãy trả lời câu hỏi của khách hàng dựa hoàn toàn trên thông tin có trong **Context** dưới đây.

Các chỉ dẫn cụ thể:
- Nếu câu hỏi yêu cầu thông tin chi tiết cho một sản phẩm cụ thể, hãy trích xuất và sử dụng các đoạn văn bản có liên quan đến sản phẩm đó.
- Nếu trong Context có nhiều sản phẩm nhưng chỉ một số sản phẩm có tên hoặc biến thể khớp với từ khóa trong câu hỏi, hãy liệt kê danh sách các sản phẩm đó kèm theo các thông tin cơ bản như tên biến thể, giá và tình trạng.
- Nếu câu hỏi yêu cầu liệt kê sản phẩm, hãy liệt kê tất cả tên sản phẩm có trong Context và mô tả ngắn gọn.
- Nếu không tìm thấy thông tin phù hợp nào trong Context, hãy trả lời: "Tôi không có thông tin về vấn đề này, xin vui lòng liên hệ bộ phận hỗ trợ khách hàng."

**Context:**
{context}

**Câu hỏi:** {input}

**Trả lời:**
""",
    input_variables=["context", "input"]
)




# --- 7. Khởi tạo LLM với retry ---
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def initialize_llm():
    try:
        llm = OllamaLLM(model="llama3", timeout=30)
        logger.info("Khởi tạo LLM thành công.")
        return llm
    except Exception as e:
        logger.error("Lỗi khi khởi tạo LLM: %s", e)
        raise

# --- Hàm in ra dữ liệu đã lưu trong vector store ---
def print_stored_data():
    persist_dir = "./chroma_phone_db"
    if os.path.exists(persist_dir):
        embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_function)
        store_data = db.get(include=["documents", "metadatas"])
        print("Tổng số tài liệu lưu trữ:", len(store_data["documents"]))
        # for i, (doc, meta) in enumerate(zip(store_data["documents"], store_data["metadatas"])):
        #     print("-" * 50)
        #     print(f"Tài liệu {i+1}:")
        #     print("Nội dung:")
        #     print(doc)
        #     print("Metadata:")
        #     print(meta)
    else:
        print("Chưa có vector store lưu trữ dữ liệu.")

# --- 8. Node xử lý chatbot ---
async def chatbot_node(state: ChatState) -> ChatState:
    chroma_dir = "./chroma_phone_db"
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    if os.path.exists(chroma_dir):
        db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)
        logger.info("Tải vector store từ %s", chroma_dir)
    else:
        logger.info("Đang xây dựng vector store mới...")
        db = await build_vector_store_async()
        if db is None:
            state["answer"] = "Không thể xây dựng vector store do không có dữ liệu."
            return state

    try:
        llm = initialize_llm()
    except Exception as e:
        logger.error("Không thể khởi tạo LLM sau 3 lần thử: %s", e)
        state["answer"] = "Không thể kết nối với Ollama. Vui lòng kiểm tra lại dịch vụ."
        return state

    max_messages = 10
    if len(state["conversation"]["messages"]) > max_messages:
        conversation_text = "\n".join(msg["content"] for msg in state["conversation"]["messages"])
        summary_prompt = f"Tóm tắt cuộc hội thoại sau thành 2-3 câu:\n{conversation_text}"
        summary = llm.invoke(summary_prompt)
        state["conversation"]["messages"] = [{"role": "system", "content": f"Tóm tắt hội thoại trước: {summary}"}]
        logger.info("Đã tóm tắt hội thoại dài.")

    retriever = db.as_retriever(search_kwargs={"k": 5})
    # In ra các tài liệu liên quan cho câu hỏi của người dùng (để debug)
    # relevant_docs = retriever.get_relevant_documents(state["question"])
    # logger.info("Số tài liệu liên quan lấy được: %d", len(relevant_docs))
    # for i, doc in enumerate(relevant_docs):
    #     logger.info("Tài liệu %d: %s", i+1, doc.page_content)
    
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)
    
    input_data = {"input": state["question"]}
    try:
        logger.info("Đang chạy chain chatbot...")
        response = chain.invoke(input_data)
        state["answer"] = response.get("answer", "Không có câu trả lời phù hợp.")
    except Exception as e:
        logger.error("Lỗi khi chạy chain chatbot: %s", e)
        state["answer"] = f"Lỗi: {str(e)}"
    
    state["conversation"]["messages"].append({"role": "user", "content": state["question"]})
    state["conversation"]["messages"].append({"role": "assistant", "content": state["answer"]})
    return state

# --- 9. Tạo graph với persistence ---
memory = MemorySaver()  
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")
app = graph.compile(checkpointer=memory)

# --- 10. Chạy chatbot ---
async def run_chatbot():
    # print("Chào bạn! Hãy đặt câu hỏi, nhập 'in_dữ_liệu' để in ra dữ liệu lưu, hoặc nhập 'quit' để thoát.")
    print("Chào bạn! Hãy đặt câu hỏi, nhập 'quit' để thoát.")

    thread_id = "user_1"
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Kết thúc hội thoại.")
            break
        # elif user_input.lower() == "in_dữ_liệu":
        #     print_stored_data()
        #     continue
        
        initial_state = ChatState(
            conversation={"messages": []},
            products={"full_product_list": "", "product_list_with_index": []},
            question=user_input,
            answer=""
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = await app.ainvoke(initial_state, config=config)
            print("Chatbot:", result["answer"])
        except Exception as e:
            logger.error("Lỗi khi chạy graph: %s", e)
            print("Chatbot: Lỗi khi xử lý câu hỏi. Vui lòng thử lại.")

# --- 11. Điểm vào chính ---
if __name__ == "__main__":
    asyncio.run(run_chatbot())
