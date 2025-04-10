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

        # Tạo danh sách Document
        documents = []
        for key, value in grouped.items():
            # in ra key
            logger.info("key: %s", key)
            # in ra value
            logger.info("value: %s", value)
            content = (
                f"{key} | Giá: {value['price']} USD | Trạng thái: {value['status']}\n"
                f"Màu sắc: {value['color']}, Dung lượng: {value['memory']}\n"
                f"{', '.join(value['attributes'])}"
            )
            documents.append(Document(page_content=content, metadata={}))
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
    retriever = db.as_retriever(search_kwargs={"k": 10})

    # Tạo chain kết hợp tài liệu với prompt
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=combine_docs_chain)

    # Kiểm tra câu hỏi có yêu cầu liệt kê chi tiết sản phẩm không
    if "hiện tại đang có bao nhiêu sản phẩm" in state["question"].lower() or "danh sách sản phẩm" in state["question"].lower():
        # Lấy tất cả tài liệu từ vector store
        all_docs = db.get(include=["documents"])["documents"]
        if not all_docs:
            state["answer"] = "Hiện tại không có sản phẩm nào trong cơ sở dữ liệu."
            return state
        
        # Loại bỏ trùng lặp
        unique_docs = list(dict.fromkeys(all_docs))
        
        # Tạo danh sách chi tiết các sản phẩm
        product_list = []
        for idx, doc in enumerate(unique_docs, 1):
            product_list.append(f"{idx}. {doc}")
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