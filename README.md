# 📱 Chatbot Tư Vấn Bán Điện Thoại

Đây là một chatbot sử dụng mô hình ngôn ngữ lớn (LLM) để tư vấn các sản phẩm điện thoại dựa trên dữ liệu từ cơ sở dữ liệu SQL Server. Ứng dụng kết hợp:

- 💬 [LangChain](https://www.langchain.com/)
- 🤗 [HuggingFace Embeddings](https://huggingface.co/)
- 🔍 [Chroma Vector Store](https://www.trychroma.com/)
- 🧠 [Ollama LLM (llama3)](https://ollama.com/)
- 🗄️ SQL Server

---

## 🚀 Cách chạy dự án

### 1. Tạo môi trường ảo

```bash
python -m venv venv
venv\Scripts\activate         # Windows
# hoặc
source venv/bin/activate     # macOS/Linux
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường `.env`

Tạo file `.env` cùng thư mục với code:

```env
DB_SERVER=localhost
DB_DATABASE=ProductDB
DB_USERNAME=sa
DB_PASSWORD=sapassword
DB_DRIVER=ODBC Driver 17 for SQL Server
```

### 4. Chạy chatbot
```bash
python chatbot_phone_consultant.py
```


### 4. Tắt chatbot
```bash
deactivate
```
---

## 🤖 Sử dụng mô hình LLM (Ollama)

### ⚠️ Ollama là gì?

Là một LLM Server chạy tại máy cục bộ, tương thích với nhiều mô hình như `llama3`, `mistral`, `gemma`,...

### 1. Cài đặt Ollama

Tải về từ: https://ollama.com/download

> Sau khi cài xong, **Ollama sẽ tự động chạy ở nền** tại `http://localhost:11434`

### 2. Pull mô hình `llama3` từ Ollama

Trước khi chạy chatbot, hãy mở terminal và chạy:

```bash
ollama pull llama3
```

> Hoặc thử các mô hình khác: `ollama pull mistral`, `ollama pull gemma`,...

---

## ▶️ Chạy chatbot

```bash
python chatbot_phone_consultant.py
```

Sau khi chạy, bạn có thể nhập câu hỏi như:

```
Bạn: Điện thoại nào có RAM 12GB và pin lớn hơn 4000mAh?
Chatbot: ...
```

---

## 🗂️ Dữ liệu

- Dữ liệu được lấy từ bảng SQL Server: `products`, `products_variants`, `product_variant_details`, `attribute_values`, `attributes`
- Sau khi load, dữ liệu sẽ được **chuyển thành văn bản** và lưu trữ dưới dạng **vector Chroma**

---

## 📦 Thư viện chính

```text
langchain
langchain-core
langchain-community
langchain-huggingface
langchain-ollama
langchain-chroma
sentence-transformers
chromadb
SQLAlchemy
python-dotenv
ollama
```

---

## ✅ Tính năng nổi bật

- Tích hợp **LangGraph** để duy trì trạng thái hội thoại
- Tự động lưu vector store vào ổ đĩa (`./chroma_phone_db`)
- Có thể kết nối với LLM miễn phí từ máy cục bộ (Ollama)

---

## 🛠️ Troubleshooting

- ❌ `ModuleNotFoundError: No module named 'langchain_ollama'`: Chạy `pip install langchain-ollama`
- ❌ `model llama3 not found`: Chưa pull -> `ollama pull llama3`
- ❌ Không kết nối được DB: Kiểm tra thông tin trong `.env`
- ❌ Không có dữ liệu: Hãy chắc chắn đã có bản ghi trong các bảng sản phẩm

---

## 📬 Liên hệ

> Nếu bạn cần hỗ trợ thêm, hãy tạo Issue hoặc liên hệ qua email.