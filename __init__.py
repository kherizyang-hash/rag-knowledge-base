"""
向量数据库模块
基于LangChain实现文档向量化存储和检索功能
"""

from vector_db_manager import VectorDatabaseManager
from document_loader import DocumentLoader
from vector_retriever import VectorRetriever, RetrievalResult, QuestionClassifier

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    "VectorDatabaseManager",
    "DocumentLoader", 
    "VectorRetriever",
    "RetrievalResult",
    "QuestionClassifier"
]

# 便捷函数
def create_vector_system(db_path: str = "./vector_databases/chroma_db",
                        embedding_model: str = "../models/text2vec-base-chinese"):
    """
    创建完整的向量检索系统
    
    Args:
        db_path: 数据库路径
        embedding_model: 嵌入模型路径
        
    Returns:
        (db_manager, retriever) 元组
    """
    db_manager = VectorDatabaseManager(
        db_path=db_path,
        embedding_model=embedding_model
    )
    
    retriever = VectorRetriever(db_manager)
    
    return db_manager, retriever


def quick_setup_academic_data(csv_path: str, 
                             db_path: str = "./vector_databases/chroma_db"):
    """
    快速设置学术数据向量数据库
    
    Args:
        csv_path: 学术数据CSV文件路径
        db_path: 数据库存储路径
        
    Returns:
        配置好的检索器
    """
    db_manager, retriever = create_vector_system(db_path=db_path)
    
    # 处理学术数据
    success = db_manager.process_csv_data(csv_path)
    
    if success:
        print(f"成功设置学术数据向量数据库: {db_path}")
        return retriever
    else:
        print(f"设置学术数据向量数据库失败")
        return None