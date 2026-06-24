"""
用途：将单个文档加载、切分、生成嵌入并上传到 Milvus 集合。
适用场景：命令行快速导入一个文件到指定集合，便于后续检索。
"""
import os
from dotenv import load_dotenv
#加载不同格式的文档
from langchain_community.document_loaders import TextLoader, PyPDFium2Loader, Docx2txtLoader, CSVLoader
#把长文本切成小块
from langchain_text_splitters import CharacterTextSplitter
#调用阿里云 API 把文字转成向量
from langchain_community.embeddings import DashScopeEmbeddings
#连接和操作 Milvus 数据库
from pymilvus import connections, Collection, CollectionSchema, DataType, FieldSchema, utility

load_dotenv()  # 读取 .env 文件中的键值对

# --- 在这里定义您的配置参数 ---
FILE_PATH = "rag.txt"  # <-- 替换为您的文件路径
#os.getenv("KEY", "默认值")：从环境变量读取值，如果不存在则用默认值
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "agent_rag")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
# ----------------------------------------

class SimpleDocumentUploader:
    #__init__ 是 Python 类的构造函数，创建对象时自动调用
    def __init__(self, host, port, collection_name, dashscope_api_key, embedding_model):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dashscope_api_key = dashscope_api_key
        self.embedding_model = embedding_model
        
        # 连接到Milvus
        self.connect_milvus()
        # 创建集合
        self.create_collection_if_not_exists()
        
    def connect_milvus(self):
        """连接到Milvus数据库"""
        # connections.connect() 是 pymilvus 提供的函数，建立与 Milvus 服务器的连接
        # connections 不是你自己写的变量 / 类，它是 pymilvus 库 内置的模块 / 对象。
        # "default" 是连接的名称（可以创建多个连接，区分不同数据库）
        connections.connect("default", host=self.host, port=self.port)
        print(f"已连接到 Milvus，地址为 {self.host}:{self.port}")
        
    def get_embedding(self, texts):
        """生成文本嵌入向量"""
        # DashScopeEmbeddings 是 LangChain 提供的类，封装了阿里云 API 调用
        embeddings_model = DashScopeEmbeddings(
            model=self.embedding_model,
            dashscope_api_key=self.dashscope_api_key
        )
        # embed_documents() 接收一个字符串列表，返回对应的向量列表
        return embeddings_model.embed_documents(texts)
        # 把所有文本都转成了向量，直接返回给调用方。

    #定义表结构。告诉 Milvus 表长什么样，就像 MySQL 的 CREATE TABLE。
    def get_schema(self):
        """定义Milvus集合的模式"""
        fields = [
            # FieldSchema 定义表中每一列。4 个字段对应 4 列：id、name（文件名）、text（原文）、embedding（向量）
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)#dim=1536：向量的维度（DashScope embedding 模型输出 1536 维）
        ]
        return CollectionSchema(fields=fields, description="文本嵌入集合")

    #创建表
    def create_collection_if_not_exists(self):
        """创建集合（如果不存在）"""
        #utility.has_collection()：检查 Milvus 中是否已有同名集合
        if not utility.has_collection(self.collection_name):
            schema = self.get_schema()
            #Collection()：创建或获取集合对象
            collection = Collection(name=self.collection_name, schema=schema)

            # {} 表示创建一个字典，用来存储「键值对（key-value pairs）」
            # 创建索引。为 embedding 字段创建索引，加速搜索
            index_params = {
                # AUTOINDEX：Milvus 自动选择最优索引类型
                # metric_type="L2"：用欧氏距离衡量相似度（距离越小越相似）
                "index_type": "AUTOINDEX",
                "metric_type": "L2",
                "params": {} # 索引的额外配置（这里是空字典，表示用默认配置）
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load() # 将集合加载到内存，准备好被搜索
            print(f"集合 '{self.collection_name}' 已创建并加载")
        else:
            print(f"集合 '{self.collection_name}' 已存在")
            
        self.collection = Collection(name=self.collection_name) # 存集合对象供后续使用
        # 前面的 collection = Collection(name=self.collection_name, schema=schema) 是在 if 代码块里定义的局部变量；
        # 这个变量 collection 只能在 if 里面用，出了 if 块就失效了，外面的 else 分支拿不到；
        # 所以你需要在 if/else 外面，统一把集合对象存到 self.collection 这个实例属性里，这样类里所有方法都能访问到它。
        
    def insert_data(self, names, texts, embeddings):
        """插入数据到Milvus"""
        # 准备数据，注意字段顺序要与schema定义一致（除了auto_id字段）
        # schema字段顺序：id(auto), name, text, embedding
        # data 是一个列表的列表，每个元素对应一列的所有行
        data = [
            names,                       # name字段  
            texts,                       # text字段
            embeddings                   # embedding字段
        ]
        
        self.collection.insert(data) # collection.insert() 批量插入
        self.collection.flush() # flush() 强制将数据刷入磁盘（确保持久化）
        print(f"已向集合插入 {len(names)} 条记录")

    # 主流程：加载→切分→向量化→插入
    def process_file(self, file_path):
        """处理文件并上传到Milvus"""
        if not os.path.exists(file_path):
            print(f"错误：文件不存在 {file_path}")
            return False
            
        # 获取文件名和扩展名
        file_name = os.path.basename(file_path)
        extension = file_name.split(".")[-1].lower()
        # split(".")：字符串方法，按 小数点 . 把字符串切割成列表。示例："abc.txt" → ["abc", "txt"]
        # [-1]：列表取最后一个元素，也就是文件后缀。示例：["abc", "txt"][-1] → "txt"
        # .lower() 字符串方法：把字母全部转为小写。
        # 作用：避免大小写问题（比如 .TXT、.Txt 统一当成 txt 判断）。

        # 根据文件类型选择加载器
        if extension == 'txt':
            # 部分指定编码 encoding='utf8' 防止乱码。
            loader = TextLoader(file_path, encoding='utf8')
        elif extension == 'pdf':
            loader = PyPDFium2Loader(file_path)
        elif extension == 'docx':
            loader = Docx2txtLoader(file_path)
        elif extension == 'csv':
            loader = CSVLoader(file_path)
        else:
            print(f"不支持的文件类型：{extension}")
            return False
            
        try:
            # 1.加载文档
            documents = loader.load()
            print(f"成功加载文档：{file_name}")
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)

            # 2.切分
            # if doc.page_content and doc.page_content.strip()：过滤条件
            # doc.page_content：文档里的正文文本；
            # strip()：去掉字符串首尾空格、换行；
            # 含义：只保留非空、不全是空白字符的有效文本；
            # doc.page_content.strip()：满足条件时，取出正文并去除首尾空白，作为列表元素。
            texts = [doc.page_content.strip() for doc in docs if doc.page_content and doc.page_content.strip()]
            # t[:2000]：截取文本前 2000 个字符；
            # 目的：防止单段文本过长，超出嵌入模型 / 向量库限制。
            final_texts = [t[:2000] for t in texts if t]

            # 3.向量化
            # 生成嵌入向量
            print("正在生成嵌入向量...")
            embeddings = self.get_embedding(final_texts)

            # 4.入库
            # 准备文件名列表
            # 列表 * 数字 语法：列表元素重复 N 次，生成新列表。
            names = [file_name] * len(final_texts)
            
            # 插入数据
            self.insert_data(names, final_texts, embeddings)
            print(f"文档 '{file_name}' 上传成功！")
            return True

        # 捕获所有通用异常，e 是异常对象，可打印错误信息
        # 防止单个文件报错导致整个程序崩溃
        except Exception as e:
            print(f"处理文件时出错：{e}")
            return False

def main():
    """主函数"""
    print("初始化文档上传器...")

    #创建 uploader 实例
    uploader = SimpleDocumentUploader(
        host=MILVUS_HOST,
        port=MILVUS_PORT,
        collection_name=COLLECTION_NAME,
        dashscope_api_key=DASHSCOPE_API_KEY,
        embedding_model=EMBEDDING_MODEL
    )

    #调用 process_file()
    print(f"开始处理文档：{FILE_PATH}")
    success = uploader.process_file(FILE_PATH)
    
    if success:
        print("文档上传完成！")
        
        # 显示集合信息
        try:
            collection = Collection(name=COLLECTION_NAME)
            count = collection.num_entities
            print(f"集合 '{COLLECTION_NAME}' 中共有 {count} 条记录")
        except Exception as e:
            print(f"获取集合信息时出错：{e}")
    else:
        print("文档上传失败！")

if __name__ == "__main__":
    main()
