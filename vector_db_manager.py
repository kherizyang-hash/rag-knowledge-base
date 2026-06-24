"""
向量数据库管理模块
基于LangChain和Milvus实现文档切分、向量化存储和检索功能
- 1.
MinIO (对象存储服务) : Milvus 使用 MinIO 来存储数据。它提供了一个网页管理界面。

- 登录页面 : http://localhost:9001
- 账号 (Access Key) : minioadmin
- 密码 (Secret Key) : minioadmin
- 2.
Milvus (向量数据库) : 在我们当前的配置中，Milvus 本身 没有 提供一个用于登录的网页界面。您可以通过代码（例如使用 pymilvus 库）连接到 Milvus 服务来进行操作。

- 连接地址 : localhost:19530
"""

import os
import json
import logging
# 记录日志	比 print 专业，可以控制日志级别（INFO/ERROR/DEBUG）
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
# 处理文件路径	比 os.path 更现代、更优雅
from dotenv import load_dotenv

# Load env vars from explicit path
env_path = Path(__file__).parent / '.env'
# 用 Path(__file__).parent 保证始终找自己旁边的 .env。
load_dotenv(dotenv_path=env_path)

# LangChain and Milvus imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings, DashScopeEmbeddings
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader, 
    CSVLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    UnstructuredExcelLoader
)
from pymilvus import utility, connections, Collection

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """向量数据库管理器 (Milvus后端)"""
    
    def __init__(self, 
                 milvus_host: str = None,
                 milvus_port: int = None,
                 collection_name: str = None,
                 embedding_model: str = None,
                 dashscope_api_key: str = None,
                 chunk_size: int = 500,
                 chunk_overlap: int = 50):
        """
        初始化向量数据库管理器
        
        Args:
            milvus_host: Milvus 服务主机
            milvus_port: Milvus 服务端口
            collection_name: Milvus 集合名称
            embedding_model: DashScope嵌入模型名称
            dashscope_api_key: DashScope API密钥
            chunk_size: 文档切分块大小
            chunk_overlap: 文档切分重叠大小
        """
        self.milvus_host = milvus_host or os.getenv("MILVUS_HOST", "127.0.0.1")
        # Ensure port is a string as pymilvus might expect it, or handle int gracefully
        self.milvus_port = str(milvus_port or os.getenv("MILVUS_PORT", "19530"))
        self.collection_name = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        self.dashscope_api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化嵌入模型
        self._init_embeddings()
        
        # 初始化文档切分器
        # RecursiveCharacterTextSplitter 的意思是：它会递归地尝试用不同的分隔符去切文本。
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        #     优先按双换行切，然后单换行，然后中文标点...优先级从高到低。尽量保持语义完整。
        )
        
        # 连接到Milvus
        self._connect_to_milvus()
        
        # 向量数据库实例
        self.vectorstore = None
        # 延迟加载：不要在 __init__ 中调用 _load_existing_db，避免启动时连接未就绪或报错
        # self._load_existing_db() 

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            # 确保 API Key 存在
            if not self.dashscope_api_key:
                 logger.warning("未提供 DashScope API Key，将尝试从环境变量获取")
                 self.dashscope_api_key = os.environ.get("DASHSCOPE_API_KEY", "")

            self.embeddings = DashScopeEmbeddings(
                model=self.embedding_model,
                dashscope_api_key=self.dashscope_api_key
            )
            # 简单测试 embedding 是否工作
            try:
                self.embeddings.embed_query("test")
                logger.info(f"成功加载并验证 DashScope嵌入模型: {self.embedding_model}")
            except Exception as e:
                 logger.error(f"DashScope 模型验证失败: {e}")
                 raise e

        except Exception as e:
            logger.error(f"加载DashScope模型失败: {e}")
            logger.warning("使用备用HuggingFace模型")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

    def _connect_to_milvus(self):
        """连接到Milvus服务"""
        try:
            logger.info(f"Connecting to Milvus: host={self.milvus_host}, port={self.milvus_port}")
            connections.connect("default", host=self.milvus_host, port=self.milvus_port)
            logger.info(f"成功连接到Milvus: {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    #     如果连接失败，raise 会把异常抛出去，让上层（调用这个方法的地方）处理。
    #     为什么不在这个方法里处理？ 因为连接失败是很严重的错误，上层可能需要决定是重试还是退出程序。
    #     这里只负责"尝试连接"，不负责"决定怎么办"。
    # 这叫"单一职责原则"：每个函数只做一件事。

    def _load_existing_db(self):
        """加载已存在的Milvus集合"""
        try:
            if utility.has_collection(self.collection_name):
                try:
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=self.collection_name,
                        connection_args={"host": self.milvus_host, "port": self.milvus_port}
                    )
                    logger.info(f"成功加载现有Milvus集合: {self.collection_name}")
                except Exception as e:
                    logger.error(f"加载现有集合失败: {e}")
                    # 不再自动删除，防止误删数据。
                    # 如果 Schema 不兼容，将在写入时处理。
                    self.vectorstore = None
            else:
                logger.info(f"未找到集合 {self.collection_name}，将在添加文档时创建")
        except Exception as e:
            logger.error(f"加载Milvus集合失败: {e}")

    def load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        # ... (代码与之前版本相同)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_extension == '.csv':
                loader = CSVLoader(file_path, encoding='utf-8')
            elif file_extension == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
            elif file_extension in ['.xlsx', '.xls']:
                loader = UnstructuredExcelLoader(file_path)
            else:
                loader = TextLoader(file_path, encoding='utf-8')
                logger.warning(f"未识别的文件类型 {file_extension}, 使用文本加载器")
            
            documents = loader.load()
            logger.info(f"成功加载文档: {file_path}, 共 {len(documents)} 个文档块")
            return documents
            
        except Exception as e:
            logger.error(f"加载文档失败 {file_path}: {e}")
            return []

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """切分文档"""
        # ... (代码与之前版本相同)
        try:
            split_docs = self.text_splitter.split_documents(documents)
            logger.info(f"文档切分完成: {len(documents)} -> {len(split_docs)} 个块")
            return split_docs
        except Exception as e:
            logger.error(f"文档切分失败: {e}")
            return documents

    def add_documents_to_db(self, documents: List[Document], collection_name: str = None):
        """
        将文档添加到Milvus数据库
        
        Args:
            documents: 文档列表
            collection_name: 集合名称（可选，覆盖默认）
        """
        if not documents:
            logger.warning("没有文档需要添加")
            return
        
        target_collection = collection_name or self.collection_name
        
        try:
            # 检查集合是否存在
            collection_exists = utility.has_collection(target_collection)
            
            if self.vectorstore is None or self.collection_name != target_collection:
                # 初始化 vectorstore
                if collection_exists:
                    logger.info(f"加载现有集合: {target_collection}")
                else:
                    logger.info(f"集合不存在，将创建新集合: {target_collection}")
                
                # Milvus.from_documents 和 Milvus(...) 的区别：
                # from_documents: 会根据文档内容创建集合（如果不存在），并插入数据。
                # Milvus(...): 仅初始化客户端，不插入数据，通常用于检索或追加。
                
                # 策略：始终使用 Milvus() 初始化，然后调用 add_documents。
                # 如果是首次创建，我们需要先确保集合存在，或者让 add_documents 处理？
                # LangChain 的 Milvus.from_documents 是最方便的初始化+插入入口。
                # 但为了避免重复创建/Schema冲突，我们应该：
                # 1. 如果集合存在，用 Milvus() 加载，然后 add_documents()
                # 2. 如果集合不存在，用 Milvus.from_documents()
                
                if collection_exists:
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"host": self.milvus_host, "port": self.milvus_port},
                        # 关键：auto_id=True 通常是默认的，但确保配置一致
                    )
                    self.collection_name = target_collection
                    # 追加数据
                    self.vectorstore.add_documents(documents)
                    logger.info(f"成功向现有集合 '{target_collection}' 追加 {len(documents)} 条文档")
                else:
                    # 集合不存在，创建并插入
                    self.vectorstore = Milvus.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"host": self.milvus_host, "port": self.milvus_port},
                        drop_old=False # 明确不删除旧的（虽然这里是else分支，本身就不存在）
                    )
                    self.collection_name = target_collection
                    logger.info(f"成功创建集合 '{target_collection}' 并插入 {len(documents)} 条文档")
            else:
                # vectorstore 已初始化且集合名称匹配，直接追加
                self.vectorstore.add_documents(documents)
                logger.info(f"成功向当前集合 '{target_collection}' 追加 {len(documents)} 条文档")
            
        except Exception as e:
            # 捕获 Schema 不兼容错误并尝试自动修复（重建）
            msg = str(e)
            # 这些错误表示表结构不匹配（比如之前的集合是用不同版本的代码创建的）。
            if "non-exist field" in msg or "inconsistent with defined schema" in msg:
                 logger.warning(f"检测到 Schema 不兼容 ({e})，尝试重建集合...")
                 try:
                     if utility.has_collection(target_collection):
                         utility.drop_collection(target_collection)
                     
                     self.vectorstore = Milvus.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"host": self.milvus_host, "port": self.milvus_port},
                        drop_old=True
                     )
                     self.collection_name = target_collection
                     logger.info(f"集合 '{target_collection}' 已重建并写入数据")
                 except Exception as re:
                     logger.error(f"重建集合失败: {re}")
                     raise re
            else:
                logger.error(f"添加文档到Milvus失败: {e}")
                raise e

    def process_file(self, file_path: str, collection_name: str = None) -> bool:
        """
        处理单个文件：加载、切分、存储
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            
        Returns:
            处理是否成功
        """
        try:
            logger.info(f"开始处理文件: {file_path}")
            documents = self.load_document(file_path)  #加载
            if not documents:
                return False
            
            split_docs = self.split_documents(documents)  #切分
            self.add_documents_to_db(split_docs, collection_name)  #存储
            
            logger.info(f"文件处理完成: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"处理文件失败 {file_path}: {e}")
            return False

    def process_csv_data(self, csv_path: str,
                         text_columns: List[str] = None,
                         metadata_columns: List[str] = None) -> bool:
        """
        处理CSV数据文件
        
        Args:
            csv_path: CSV文件路径
            text_columns: 需要向量化的文本列名列表
            
        Returns:
            处理是否成功
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, encoding='utf-8')
            logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
            if text_columns is None or not text_columns:
                object_cols = [c for c in df.columns if df[c].dtype == 'object']
                text_columns = [c for c in object_cols if not str(c).startswith('Unnamed')]

            if metadata_columns is None:
                metadata_columns = [c for c in df.columns if c not in text_columns]

            documents = []
            for idx, row in df.iterrows():
                content_parts = []
                metadata = {"source": csv_path, "row_index": idx}
                
                for col in text_columns:
                    if pd.notna(row[col]):
                        text = str(row[col]).strip()
                        if text:
                            content_parts.append(f"{col}: {text}")

                for col in metadata_columns:
                    val = row.get(col)
                    if pd.notna(val):
                        metadata[str(col)] = str(val)
                
                if content_parts:
                    content = "\n".join(content_parts)
                    doc = Document(page_content=content, metadata=metadata)
                    documents.append(doc)
            
            logger.info(f"构建了 {len(documents)} 个文档")
            split_docs = self.split_documents(documents)
            self.add_documents_to_db(split_docs)
            
            return True
            
        except Exception as e:
            logger.error(f"处理CSV数据失败: {e}")
            return False

    def get_embedding(self, texts: List[str]) -> List[List[float]]:
        """使用嵌入模型为一组文本生成嵌入向量"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"生成嵌入向量失败: {e}")
            return []

    def search(self, query: str, k: int = 5, filter_dict: Dict = None, collection_name: Optional[str] = None) -> List[Tuple[Document, float]]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter_dict: Milvus目前不支持直接的元数据过滤，此参数保留但暂不使用
            collection_name: 指定要搜索的集合名称（可选）
        
        Returns:
            (文档, 相似度分数) 列表
        """
        target_collection = collection_name or self.collection_name

        if self.vectorstore is None or (target_collection and self.collection_name != target_collection):
            try:
                if target_collection and utility.has_collection(target_collection):
                    self.vectorstore = Milvus(
                        embedding_function=self.embeddings,
                        collection_name=target_collection,
                        connection_args={"host": self.milvus_host, "port": self.milvus_port}
                    )
                    self.collection_name = target_collection
                    logger.info(f"加载集合用于搜索: {target_collection}")
                else:
                    logger.warning("向量数据库未初始化")
                    return []
            except Exception as e:
                logger.error(f"加载Milvus集合失败: {e}")
                return []
        
        try:
            if filter_dict:
                logger.warning("Milvus集成当前不支持直接的元数据过滤，该过滤器将被忽略。")
            
            results = self.vectorstore.similarity_search_with_score(query=query, k=k)
            logger.info(f"搜索查询: '{query}', 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def get_database_info(self, collection_name: str = None) -> Dict[str, Any]:
        """
        获取数据库信息。可以用来监控数据库状态。
        """
        # 提供了 get_database_info 方法，用于获取数据库的实时统计信息。
        # 这在运维监控和调试时非常重要，可以快速了解当前知识库有多少条数据。
        target_collection = collection_name or self.collection_name
        
        info = {
            "milvus_host": self.milvus_host,
            "milvus_port": self.milvus_port,
            "collection_name": target_collection,
            "is_initialized": self.vectorstore is not None
        }
        
        try:
            # 如果 vectorstore 未初始化，尝试临时连接检查
            if utility.has_collection(target_collection):
                # 使用 Collection 对象获取统计信息
                col = Collection(target_collection)
                # 刷新以确保获取最新数据量（刚插入的数据可能还在内存中）
                col.flush()  #强制把内存中的数据刷到磁盘，确保 num_entities 是最新的。
                info["document_count"] = col.num_entities
                
                # 如果 self.vectorstore 为空但集合存在，尝试初始化它（如果还没做过）
                if self.vectorstore is None:
                     try:
                        self.vectorstore = Milvus(
                            embedding_function=self.embeddings,
                            collection_name=target_collection,
                            connection_args={"host": self.milvus_host, "port": self.milvus_port}
                        )
                        info["is_initialized"] = True
                     except:
                        pass # 忽略加载错误，只返回统计
            else:
                info["document_count"] = 0
        except Exception as e:
            logger.error(f"获取Milvus集合信息失败: {e}")
            info["error"] = str(e)
        
        return info

    def clear_database(self):
        """清空Milvus集合"""
        try:
            if utility.has_collection(self.collection_name):
                utility.drop_collection(self.collection_name)
                self.vectorstore = None
                logger.info(f"Milvus集合 '{self.collection_name}' 已被删除")
        except Exception as e:
            logger.error(f"清空Milvus集合失败: {e}")

def main():
    """测试函数"""
    # 确保Milvus服务正在运行
    try:
        connections.connect("default", host="127.0.0.1", port=19530)
        utility.get_collection_stats("agent_rag")
        connections.disconnect("default")
    except Exception as e:
        logger.error("无法连接到Milvus服务，请确保您已通过 docker-compose up -d 启动了Milvus。")
        logger.error(f"错误: {e}")
        return

    # 创建向量数据库管理器
    db_manager = VectorDatabaseManager()
    
    # 清空现有数据
    print("清空现有数据库...")
    db_manager.clear_database()

    # 处理学科数据CSV文件
    csv_path = "../data.csv"
    if os.path.exists(csv_path):
        print("处理CSV数据...")
        success = db_manager.process_csv_data(csv_path)
        if success:
            print("CSV数据处理成功！")
            
            # 测试搜索
            print("\n测试搜索功能:")
            results = db_manager.search("示例查询", k=3)
            for i, (doc, score) in enumerate(results):
                print(f"\n结果 {i+1} (相似度: {score:.4f}):")
                print(f"内容: {doc.page_content[:200]}...")
                print(f"元数据: {doc.metadata}")
        else:
            print("CSV数据处理失败！")
    else:
        print(f"未找到数据文件: {csv_path}")
    
    # 显示数据库信息
    print("\n数据库信息:")
    info = db_manager.get_database_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
