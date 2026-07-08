import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
# 核心修正：从全新的标准库中导入切分器
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

print("正在初始化，准备读取知识库...")

# 1. 配置路径
KB_DIR = "./knowledge_base"
DB_DIR = "./chroma_db"

# 2. 自动遍历读取所有子文件夹中的 .txt 文件
loader = DirectoryLoader(KB_DIR, glob="**/*.txt", loader_cls=lambda path: TextLoader(path, encoding='utf-8'))
documents = loader.load()
print(f"成功读取到 {len(documents)} 个原始故障案例文件。")

# 3. 将文本切分成适合 AI 检索的小片段
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=40)
docs = text_splitter.split_documents(documents)
print(f"文本切分完成，共生成 {len(docs)} 个知识片段。")

# 4. 加载本地向量嵌入模型
print("正在加载向量嵌入模型，请稍候...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 5. 构建并保存向量数据库
print("正在将文字转化为向量并写入硬盘...")
db = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)

print("\n恭喜！本地故障向量数据库构建成功！")
print(f"数据库已保存在：{os.path.abspath(DB_DIR)}")
