import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
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

# --- Cấu hình ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === 1. Hàm truy vấn dữ liệu từ SQL Server ===
def fetch_product_documents():
    try:
        server = os.getenv("DB_SERVER", "localhost")
        database = os.getenv("DB_DATABASE", "ProductDB")
        username = os.getenv("DB_USERNAME", "sa")
        password = os.getenv("DB_PASSWORD", "sapassword")
        driver = os.getenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
        conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver.replace(' ', '+')}"
        engine = create_engine(conn_str)
        query = text("""
            SELECT 
                p.name AS product_name,
                pv.name AS variant_name,
                pv.name + ' - ' + c.name + ' - ' + m.name AS pvd_name,
                pvd.price,
                pvd.status,
                c.name AS color_name,
                m.name AS memory_name,
                a.name AS attribute_name,
                av.value AS attribute_value
            FROM products p
            JOIN products_variants pv ON p.id = pv.product_id
            JOIN product_variant_details pvd ON pv.id = pvd.product_variant_id
            JOIN attribute_values av ON av.product_variant_id = pv.id
            JOIN attributes a ON av.attribute_id = a.id
            JOIN colors c ON c.id = pvd.color_id
            JOIN memories m ON pvd.memory_id = m.id
        """)
        with engine.connect() as conn:
            result = conn.execute(query).mappings().all()
        
        # Gom nhóm dữ liệu theo product và variant
        grouped = {}
        for row in result:
            key = f"{row['product_name']} - {row['pvd_name']}"
            if key not in grouped:
                grouped[key] = {
                    "price": row["price"],
                    "status": row["status"],
                    "color": row["color_name"],
                    "memory": row["memory_name"],
                    "attributes": []
                }
            grouped[key]["attributes"].append(f"{row['attribute_name']}: {row['attribute_value']}")
        
        # Log dữ liệu grouped để kiểm tra
        for key, value in grouped.items():
            logger.info("key: %s", key)
            logger.info("value: %s", value)

        # Tạo danh sách Document
        documents = []
        for key, value in grouped.items():
            product_name = key.split(" - ")[0]  # Lấy p.name
            pvd_name = key[len(product_name) + 3:]  # Lấy pvd_name (bỏ " - " giữa p.name và pvd_name)
            content = (
                f"{pvd_name} | Giá: {value['price']} VND | Trạng thái: {value['status']}\n"
                f"Màu sắc: {value['color']}, Dung lượng: {value['memory']}\n"
                f"{', '.join(value['attributes'])}"
            )
            documents.append(Document(page_content=content, metadata={"product_name": product_name}))
        logger.info("Danh sách tài liệu: %s", documents)
        return documents
    except Exception as e:
        logger.error("Lỗi khi lấy dữ liệu sản phẩm: %s", e)
        return []

# === 2. Xây dựng vector store ===
def build_vector_store():
    docs = fetch_product_documents()
    if not docs:
        logger.warning("Không tìm thấy tài liệu từ cơ sở dữ liệu.")
        return None
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Làm sạch vector store trước khi xây dựng lại
    persist_dir = "./chroma_phone_db"
    if os.path.exists(persist_dir):
        logger.info("Xóa vector store cũ trước khi xây dựng mới...")
        import shutil
        shutil.rmtree(persist_dir)
    
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding_function, persist_directory=persist_dir)
    return db

# === 3. Định nghĩa trạng thái cho LangGraph ===
class ChatState(TypedDict):
    messages: List[Dict[str, str]]  # Lưu lịch sử hội thoại [{"role": "user/assistant", "content": "..."}]
    question: str                   # Câu hỏi hiện tại
    answer: str                     # Câu trả lời từ chatbot

# === 4. Prompt chatbot ===
prompt_template = PromptTemplate(
    template="""
Bạn là một chuyên viên tư vấn bán điện thoại thông minh.
Trả lời các câu hỏi của khách hàng dựa trên thông tin có sẵn.
Nếu câu hỏi liên quan, hãy sử dụng tài liệu trong phần "Context".
Nếu bạn không chắc chắn, vui lòng nói "Tôi không chắc, xin vui lòng liên hệ bộ phận hỗ trợ khách hàng."
Trả lời ngắn gọn, rõ ràng và chuyên nghiệp.

Context:
{context}

Câu hỏi: {input}
Trả lời:
""",
    input_variables=["context", "input"]
)

# === 5. Node xử lý chatbot ===
def chatbot_node(state: ChatState) -> ChatState:
    chroma_dir = "./chroma_phone_db"
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Tải hoặc xây dựng vector store
    if os.path.exists(chroma_dir):
        db = Chroma(persist_directory=chroma_dir, embedding_function=embedding_function)
        logger.info("Tải vector store từ %s", chroma_dir)
    else:
        logger.info("Đang xây dựng vector store mới...")
        db = build_vector_store()
        if db is None:
            state["answer"] = "Không thể xây dựng vector store do không có dữ liệu."
            return state

    # Khởi tạo LLM
    try:
        llm = OllamaLLM(model="llama3", timeout=30)
    except Exception as e:
        logger.error("Lỗi khi kết nối với Ollama: %s", e)
        state["answer"] = "Không thể kết nối với Ollama. Vui lòng kiểm tra xem dịch vụ Ollama đã được khởi động chưa."
        return state

    # Tạo retriever
    retriever = db.as_retriever(search_kwargs={"k": 20})  # Tăng k để lấy nhiều tài liệu hơn

    # Tạo chain kết hợp tài liệu với prompt
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

    # Kiểm tra câu hỏi có yêu cầu liệt kê chi tiết sản phẩm không
    liet_ke_san_pham_keywords = [
        "hiện tại có bao nhiêu sản phẩm",
        "danh sách sản phẩm",
        "liệt kê sản phẩm",
        "có những sản phẩm nào",
        "tổng số sản phẩm",
        "bao nhiêu mẫu điện thoại",
        "sản phẩm đang bán",
        "sản phẩm có sẵn",
        "hiển thị sản phẩm",
        "liệt kê danh sách điện thoại",
    ]

    if any(k in state["question"].lower() for k in liet_ke_san_pham_keywords):

        # Lấy tất cả tài liệu từ vector store
        all_docs_data = db.get(include=["documents", "metadatas"])
        all_docs = all_docs_data["documents"]
        metadatas = all_docs_data["metadatas"]
        if not all_docs:
            state["answer"] = "Hiện tại không có sản phẩm nào trong cơ sở dữ liệu."
            return state
        
        # Log dữ liệu từ vector store để kiểm tra
        logger.info("Tổng số tài liệu từ vector store: %d", len(all_docs))
        logger.info("Danh sách tài liệu từ vector store: %s", all_docs)

        # Loại bỏ trùng lặp
        unique_docs = []
        seen = set()
        for doc, meta in zip(all_docs, metadatas):
            doc_key = doc.split("\n")[0]  # Dùng dòng đầu tiên (pvd_name) làm khóa
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append((doc, meta["product_name"]))

        # Log sau khi loại bỏ trùng lặp
        logger.info("Tổng số tài liệu sau khi loại bỏ trùng lặp: %d", len(unique_docs))

        # Nhóm các variant theo sản phẩm
        grouped_products = {}
        for doc, product_name in unique_docs:
            if product_name not in grouped_products:
                grouped_products[product_name] = []
            grouped_products[product_name].append(doc)

        # Tạo danh sách chi tiết các sản phẩm
        product_list = []
        idx = 1
        for product_name, variants in grouped_products.items():
            product_list.append(f"{product_name}:")
            for variant in variants:
                product_list.append(f"  {idx}. {variant}")
                idx += 1

        response_text = f"Hiện tại có {len(unique_docs)} sản phẩm:\n" + "\n".join(product_list)
        state["answer"] = response_text
    else:
        # Xử lý các câu hỏi khác bằng chain
        input_data = {"input": state["question"]}
        try:
            response = chain.invoke(input_data)
            state["answer"] = response.get("answer", "Không có câu trả lời phù hợp.")
        except Exception as e:
            logger.error("Lỗi khi chạy chain chatbot: %s", e)
            state["answer"] = f"Lỗi: {str(e)}"

    # Cập nhật lịch sử hội thoại
    state["messages"].append({"role": "user", "content": state["question"]})
    state["messages"].append({"role": "assistant", "content": state["answer"]})
    return state

# === 6. Tạo graph với persistence ===
memory = MemorySaver()  # Lưu trữ trong RAM
graph = StateGraph(ChatState)
graph.add_node("chatbot", chatbot_node)
graph.set_entry_point("chatbot")
graph.set_finish_point("chatbot")
app = graph.compile(checkpointer=memory)

# === 7. Chạy chatbot ===
if __name__ == "__main__":
    print("Chào bạn! Hãy đặt câu hỏi, hoặc nhập 'quit' để thoát.")
    thread_id = "user_1"  # ID để theo dõi hội thoại của người dùng
    while True:
        user_input = input("Bạn: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Kết thúc hội thoại.")
            break
        
        # Khởi tạo trạng thái ban đầu
        initial_state = ChatState(messages=[], question=user_input, answer="")
        
        # Chạy graph với persistence
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = app.invoke(initial_state, config=config)
            print("Chatbot:", result["answer"])
        except Exception as e:
            logger.error("Lỗi khi chạy graph: %s", e)
            print("Chatbot: Lỗi khi xử lý câu hỏi. Vui lòng thử lại.")