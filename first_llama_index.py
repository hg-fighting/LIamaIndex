from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.dashscope import DashScope
from llama_index.embeddings.dashscope import DashScopeEmbedding

# 配置千问（按官方文档的思路：设置全局 LLM + Embedding）
Settings.llm = DashScope(
    model_name="qwen-turbo",
    api_key="sk-1a508fab8ba34804add68f62883d1470"
)

Settings.embed_model = DashScopeEmbedding(
    model_name="text-embedding-v2",
    api_key="sk-1a508fab8ba34804add68f62883d1470"
)

# 加载数据（官方文档这行一模一样）
documents = SimpleDirectoryReader("data").load_data()

# 构建索引（官方文档核心代码）
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 提问
response = query_engine.query("What is AI?")
print(response)