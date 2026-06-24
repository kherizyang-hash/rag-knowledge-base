import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

# from example.vector_databases.api_integration import register_vector_routes
from api_integration import register_vector_routes

# 添加当前目录到 python path 以便导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # 启用跨域支持，允许前端访问
    
    # 注册向量数据库路由
    register_vector_routes(app)
    
    @app.route('/')
    def index():
        return "RAG Backend Service is Running!"

    return app

if __name__ == '__main__':
    app = create_app()
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print("启动 RAG 后端服务...")
    print(f"API 地址: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/api/vector/")
    app.run(host=host, port=port, debug=debug)
