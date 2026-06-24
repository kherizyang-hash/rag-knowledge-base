"""
API集成模块 - 使用新版 MilvusClient
将向量数据库功能集成到Flask应用中
"""
# Flask 的蓝图，用来组织路由。你可以把不同功能的 API 分到不同蓝图，然后在主应用里注册。
# 这样代码更整洁，不会把几百个 API 塞在一个文件里。
# request：Flask 提供的对象，用来获取 HTTP 请求里的数据（JSON、文件等）。
# jsonify：把 Python 字典转成 JSON 格式的 HTTP 响应。
from flask import Blueprint, request, jsonify
import os
import logging
from typing import Dict, Any, List
# secure_filename：一个安全工具。用户上传的文件名可能包含 ../../ 之类的路径遍历攻击，
# 这个函数会过滤掉危险字符，只保留安全的文件名。
from werkzeug.utils import secure_filename
from pathlib import Path
from dotenv import load_dotenv
from pymilvus import MilvusClient
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader, CSVLoader
from openai import OpenAI

env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# 创建蓝图
vector_bp = Blueprint('vector', __name__, url_prefix='/api/vector')
# url_prefix='/api/vector'：这个蓝图下的所有路由都会自动加上这个前缀。
# 比如你定义 @vector_bp.route('/query')，实际访问路径就是 /api/vector/query。

# 全局变量
milvus_client: MilvusClient = None
embeddings_model = None
llm_client = None

# 临时上传目录
UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_vector_system(
    milvus_host: str = None,
    milvus_port: str = None,
    embedding_model: str = None,
    dashscope_api_key: str = None,
    collection_name: str = None
):
    """初始化向量系统（使用新版 MilvusClient）"""
    global milvus_client, embeddings_model, llm_client

    try:
        # 获取配置
        host = milvus_host or os.getenv("MILVUS_HOST", "localhost")
        port = milvus_port or os.getenv("MILVUS_PORT", "19530")
        api_key = dashscope_api_key or os.getenv("DASHSCOPE_API_KEY", "")
        model_name = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-v1")
        collection = collection_name or os.getenv("COLLECTION_NAME", "agent_rag")

        # 1. 连接 Milvus（新版 Client）
        milvus_uri = f"http://{host}:{port}"
        milvus_client = MilvusClient(uri=milvus_uri)

        # 2. 检查/创建集合
        if not milvus_client.has_collection(collection):
            milvus_client.create_collection(
                collection_name=collection,
                dimension=1536,
                metric_type="L2",
                auto_id=True
            )
            logger.info(f"创建新集合: {collection}")
        else:
            logger.info(f"集合已存在: {collection}")

        # 3. 初始化嵌入模型
        embeddings_model = DashScopeEmbeddings(
            model=model_name,
            dashscope_api_key=api_key
        )

        # 4. 初始化 LLM 客户端
        llm_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        logger.info(f"向量系统初始化成功，Milvus: {milvus_uri}, 集合: {collection}")
        return True

    except Exception as e:
        logger.error(f"向量系统初始化失败: {str(e)}")
        return False


def get_collection_name(collection_name: str = None) -> str:
    """获取集合名"""
    return collection_name or os.getenv("COLLECTION_NAME", "agent_rag")


def load_document(file_path: str):
    """根据文件类型加载文档"""
    extension = Path(file_path).suffix.lower()

    if extension == '.txt':
        loader = TextLoader(file_path, encoding='utf-8')
    elif extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif extension == '.docx':
        loader = Docx2txtLoader(file_path)
    elif extension == '.csv':
        loader = CSVLoader(file_path, encoding='utf-8')
    else:
        loader = TextLoader(file_path, encoding='utf-8')
    #     如果扩展名不识别，它会兜底用 TextLoader，而不是直接报错。

    return loader.load()


@vector_bp.route('/upload_file', methods=['POST'])
def upload_file():
    """上传文件流处理"""
    global milvus_client, embeddings_model

    if milvus_client is None:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': '未找到文件'}), 400

    file = request.files['file']
    collection_name = request.form.get('collection_name', get_collection_name())

    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择文件'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)

        try:
            # 加载文档
            documents = load_document(file_path)

            # 切分文档
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(documents)
            texts = [doc.page_content.strip() for doc in split_docs if doc.page_content.strip()]

            if not texts:
                return jsonify({'success': False, 'message': '文档内容为空'}), 400

            # 生成向量
            vectors = embeddings_model.embed_documents(texts)

            # 准备插入数据
            data = []
            for text, vector in zip(texts, vectors):
                data.append({
                    "text": text,
                    "name": filename,
                    "embedding": vector
                })

            # 插入 Milvus
            insert_result = milvus_client.insert(
                collection_name=collection_name,
                data=data
            )

            # 清理临时文件
            os.remove(file_path)

            # 获取统计信息
            stats = milvus_client.get_collection_stats(collection_name)

            return jsonify({
                'success': True,
                'message': f'文件上传成功: {filename}',
                'insert_count': insert_result['insert_count'],
                'total_count': stats['row_count']
            })

        except Exception as e:
            logger.error(f"文件处理失败: {e}")
            return jsonify({'success': False, 'message': f'文件处理失败: {str(e)}'}), 500

    return jsonify({'success': False, 'message': '上传失败'}), 500


@vector_bp.route('/query', methods=['POST'])
def query_documents():
    """RAG 问答"""
    global milvus_client, embeddings_model, llm_client

    if milvus_client is None:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({'success': False, 'message': '请提供 question 参数'}), 400

        question = data['question']
        collection_name = data.get('collection_name', get_collection_name())
        k = data.get('k', 5)

        # 1. 问题向量化
        query_vector = embeddings_model.embed_documents([question])[0]

        # 2. 向量搜索
        search_result = milvus_client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=k,
            output_fields=["text", "name"]
        )

        # 3. 提取检索到的文本
        contexts = []
        source_docs = []
        scores = []
        for result in search_result[0]:
            text = result['entity']['text']
            score = result['distance']
            name = result['entity'].get('name', 'unknown')
            contexts.append(text)
            source_docs.append({"content": text, "source": name, "score": score})
            scores.append(score)

        # 4. 构建 Prompt 并调用 LLM
        if contexts:
            context_str = "\n\n".join([f"参考资料{i+1}: {ctx}" for i, ctx in enumerate(contexts)])
            system_prompt = "你是一个智能助手。请基于【参考资料】回答用户的问题。如果参考资料与问题无关，请说明。"
            user_prompt = f"问题：{question}\n\n【参考资料】\n{context_str}"
        else:
            system_prompt = "你是一个智能助手。请用你的通用知识回答用户的问题。"
            user_prompt = question

        response = llm_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "qwen-plus"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )

        answer = response.choices[0].message.content

        # 5. 计算置信度（简单版）
        confidence = min(max(scores) / 20000 if scores else 0, 1.0)

        return jsonify({
            'success': True,
            'question': question,
            'answer': answer,
            'confidence': confidence,
            'sources': source_docs
        })

    except Exception as e:
        logger.error(f"查询API错误: {str(e)}")
        return jsonify({'success': False, 'message': f'查询失败: {str(e)}'}), 500


@vector_bp.route('/collection_info', methods=['GET'])
def get_collection_info():
    """获取集合信息"""
    global milvus_client

    if milvus_client is None:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    collection_name = request.args.get('collection_name', get_collection_name())

    try:
        if milvus_client.has_collection(collection_name):
            stats = milvus_client.get_collection_stats(collection_name)
            return jsonify({
                'success': True,
                'collection_name': collection_name,
                'row_count': stats['row_count']
            })
        else:
            return jsonify({
                'success': True,
                'collection_name': collection_name,
                'row_count': 0,
                'message': '集合不存在'
            })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@vector_bp.route('/clear_collection', methods=['POST'])
def clear_collection():
    """清空集合"""
    global milvus_client

    if milvus_client is None:
        return jsonify({'success': False, 'message': '向量系统未初始化'}), 400

    data = request.get_json()
    collection_name = data.get('collection_name', get_collection_name()) if data else get_collection_name()

    try:
        if milvus_client.has_collection(collection_name):
            milvus_client.drop_collection(collection_name)
            # 重建空集合
            milvus_client.create_collection(
                collection_name=collection_name,
                dimension=1536,
                metric_type="L2",
                auto_id=True
            )
            return jsonify({'success': True, 'message': f'集合 {collection_name} 已清空'})
        else:
            return jsonify({'success': False, 'message': f'集合 {collection_name} 不存在'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


def register_vector_routes(app):
    """注册向量数据库路由到Flask应用"""
    app.register_blueprint(vector_bp)
    
    # 自动初始化向量系统
    with app.app_context():
        init_vector_system()
    
    logger.info("向量数据库API路由已注册")