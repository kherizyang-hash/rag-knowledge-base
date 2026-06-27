
# RAG 知识库助手

基于 **Milvus + LangChain + DashScope** 构建的 RAG（检索增强生成）知识库问答系统。支持上传 PDF、DOCX、TXT、CSV 等格式文档，自动切分、向量化并存入向量数据库，用户提问时系统检索相关文档片段并调用大语言模型生成答案。

## 效果展示
Web 端采用左右分栏布局：左侧为知识库配置与文档入库，右侧为智能问答。以下截图基于 `rag.txt` 示例文档演示完整流程。
### 界面概览
进入系统后，可在左侧上传文档、配置集合名称，在右侧进行自然语言提问。
![界面概览](docs/screenshots/屏幕截图%202026-06-27%20100351.png)
### 文档入库
支持拖拽或点击上传，解析完成后提示「入库成功」，文档内容即写入 Milvus 指定集合。
![文档入库](docs/screenshots/屏幕截图%202026-06-27%20103402.png)
### 智能问答
基于向量检索召回相关片段，由大模型生成结构化回答，内容紧扣已上传的知识库。
![智能问答](docs/screenshots/屏幕截图%202026-06-27%20103450.png)
### 参考依据溯源
每条回答下方可展开「参考依据」，查看检索到的原文片段及相似度分数，便于核对答案来源。
![参考依据溯源](docs/screenshots/屏幕截图%202026-06-27%20104643.png)

## 功能特性

- 多格式文档上传（PDF、DOCX、TXT、CSV）
- 文档自动切分与向量化
- 向量相似度检索（Milvus + L2 距离）
- RAG 智能问答（通义千问 Qwen-Plus）
- REST API（Flask）+ Web 前端（Vue3）
- 答案引用来源可追溯


## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 向量数据库 | Milvus + etcd + MinIO |
| RAG 框架 | LangChain |
| 嵌入模型 | DashScope text-embedding-v1 |
| 大语言模型 | 通义千问 Qwen-Plus |
| Web 框架 | Flask + Flask-CORS |
| 前端 | Vue3 + Vite |
| 文档解析 | PyPDF, Docx2txt, CSVLoader |


## 项目结构

```
.
├── api_integration.py      # Flask API 路由
├── server.py               # Flask 服务入口
├── vector_db_manager.py    # Milvus 向量数据库管理
├── vector_retriever.py     # 检索与问答逻辑
├── document_loader.py      # 多格式文档加载
├── upload_document.py      # 命令行上传脚本
├── query_system.py         # 命令行问答脚本
├── requirements.txt        # Python 依赖
├── docker-compose.yml      # Milvus 容器编排
├── .env.example            # 环境变量模板
├── rag_front/              # Vue3 前端源码
└── test/                   # 测试脚本
```


## 快速启动

### 1. 环境准备

- Python 3.9+
- Docker Desktop
- 阿里云 DashScope API Key（[获取地址](https://dashscope.aliyun.com/)）

### 2. 克隆项目

```bash
git clone https://github.com/kherizyang-hash/rag-knowledge-base.git
cd rag-knowledge-base
```

### 3. 创建并激活虚拟环境

**Windows（CMD）**：
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows（PowerShell）**：
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**：
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. 安装依赖

```bash
pip install -r requirements.txt
```

### 5. 配置环境变量

复制 `.env.example` 为 `.env`：

```bash
copy .env.example .env
```

然后编辑 `.env`，填入你的 API Key：

```ini
DASHSCOPE_API_KEY=sk-你的API密钥
```

### 6. 启动 Milvus

在项目根目录执行：

```bash
docker-compose up -d
```

验证容器正常运行：

```bash
docker ps
```

### 7. 启动后端服务

```bash
python server.py
```

服务启动后，API 地址为 `http://localhost:5000/api/vector/`

### 8. 测试上传文档（命令行）

```bash
python upload_document.py
```

### 9. 测试问答（命令行）

```bash
python query_system.py
```

### 10. 启动前端（可选）

```bash
cd rag_front
npm install
npm run dev
```

访问 `http://localhost:5173` 即可使用 Web 界面。


## API 使用示例

### 上传文档

```bash
curl -X POST http://localhost:5000/api/vector/upload_file \
  -F "file=@文档.pdf" \
  -F "collection_name=agent_rag"
```

### 问答

```bash
curl -X POST http://localhost:5000/api/vector/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "collection_name": "agent_rag", "k": 5}'
```


## 工程化特性

- 嵌入模型健康检查，启动时验证 API Key 有效性
- DashScope API 失败时自动降级到本地 HuggingFace 模型
- Milvus 集合懒加载，按需连接
- Schema 冲突自动检测与重建
- 防御性编程，多处配置校验


## 常见问题

**Q: 提示 Milvus 连接失败？**
- 确认 Docker Desktop 已启动
- 确认已执行 `docker-compose up -d`
- 等待 30-60 秒让容器完全初始化

**Q: 提示 API Key 无效？**
- 检查 `.env` 中的 `DASHSCOPE_API_KEY` 是否正确
- 确认阿里云账号已开通模型服务


## 许可证

MIT License


## 作者

kherizyang-hash
