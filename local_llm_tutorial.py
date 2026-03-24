from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 加载本地模型 —— 修复版！
Settings.llm = HuggingFaceLLM(
    model_name="Qwen/Qwen-7B-Chat",
    device_map="auto",
    # 下面这两个是修复报错的关键！
    model_kwargs={"trust_remote_code": True},
    generate_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# 本地 Embedding（你一直在用的 m3e）
Settings.embed_model = HuggingFaceEmbedding(
    model_name="moka-ai/m3e-base",
    device="cpu"
)

# 加载文档（官方教程写法）
documents = SimpleDirectoryReader("data").load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 提问
query_engine = index.as_query_engine()
response = query_engine.query("What is AI?")

print("\n✅ 回答：")
print(response)