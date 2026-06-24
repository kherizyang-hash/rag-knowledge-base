<template>
  <div class="app-layout">
    <!-- 侧边栏 -->
    <aside class="sidebar">
      <div class="logo-area">
        <div class="logo-icon">🧠</div>
        <h1>RAG 智能助手</h1>
      </div>

      <div class="sidebar-content">
        <div class="section-title">知识库设置</div>
        
        <div class="upload-box">
          <el-upload
            class="upload-dragger"
            drag
            action="#"
            :auto-upload="false"
            :on-change="handleFileChange"
            :show-file-list="true"
            :limit="1"
            :on-exceed="handleExceed"
            ref="uploadRef"
          >
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="upload-text">
              <p>点击或拖拽文件至此</p>
              <span>支持 PDF, DOCX, TXT, CSV</span>
            </div>
          </el-upload>
        </div>

        <div class="form-group">
          <label>集合名称</label>
          <el-input 
            v-model="collectionName" 
            placeholder="例如: agent_rag" 
            class="custom-input"
          >
            <template #prefix>
              <el-icon><Collection /></el-icon>
            </template>
          </el-input>
        </div>

        <el-button 
          type="primary" 
          class="action-btn upload-btn" 
          :loading="uploading" 
          @click="submitUpload"
        >
          {{ uploading ? '正在解析入库...' : '立即解析入库' }}
        </el-button>

        <div v-if="uploadSuccess" class="success-tip">
          <el-icon><CircleCheckFilled /></el-icon>
          <span>入库成功，快去提问吧！</span>
        </div>
      </div>

      <div class="sidebar-footer">
        <p>© 2024 MCP Agent System</p>
      </div>
    </aside>

    <!-- 主聊天区域 -->
    <main class="chat-main">
      <!-- 顶部导航 -->
      <header class="chat-header">
        <div class="header-info">
          <h2>智能问答</h2>
          <span class="status-badge">
            <span class="status-dot"></span>
            当前集合: {{ collectionName }}
          </span>
        </div>
        <div class="header-actions">
          <el-button circle icon="Delete" @click="clearHistory" title="清空对话" />
        </div>
      </header>

      <!-- 消息列表 -->
      <div class="messages-container" ref="messagesRef">
        <div v-if="messages.length === 0" class="welcome-screen">
          <div class="welcome-icon">👋</div>
          <h3>你好！我是你的专属知识助手</h3>
          <p>请在左侧上传文档，然后在这里向我提问。</p>
          <div class="feature-list">
            <div class="feature-item">
              <el-icon><Document /></el-icon>
              <span>文档解析</span>
            </div>
            <div class="feature-item">
              <el-icon><Search /></el-icon>
              <span>精准检索</span>
            </div>
            <div class="feature-item">
              <el-icon><ChatLineRound /></el-icon>
              <span>智能问答</span>
            </div>
          </div>
        </div>

        <transition-group name="message-fade">
          <div 
            v-for="(msg, index) in messages" 
            :key="index" 
            :class="['message-row', msg.role]"
          >
            <div class="avatar">
              <el-avatar :size="40" :icon="msg.role === 'user' ? User : Service" :class="msg.role" />
            </div>
            <div class="message-content">
              <div class="bubble" v-html="renderMarkdown(msg.content)"></div>
              
              <!-- 引用来源卡片 -->
              <div v-if="msg.sources && msg.sources.length > 0" class="sources-card">
                <div class="sources-header" @click="toggleSources(index)">
                  <el-icon><CollectionTag /></el-icon>
                  <span>参考依据 ({{ msg.sources.length }})</span>
                  <el-icon :class="['arrow', { rotated: msg.showSources }]"><ArrowDown /></el-icon>
                </div>
                <div v-show="msg.showSources" class="sources-list">
                  <div v-for="(source, sIndex) in msg.sources" :key="sIndex" class="source-item">
                    <div class="source-meta">
                      <span class="index">#{{ sIndex + 1 }}</span>
                      <span class="score">相似度: {{ (source.score * 100).toFixed(1) }}%</span>
                    </div>
                    <div class="source-text">{{ source.content }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </transition-group>

        <div v-if="thinking" class="message-row assistant thinking-row">
          <div class="avatar">
            <el-avatar :size="40" :icon="Service" class="assistant" />
          </div>
          <div class="message-content">
            <div class="bubble thinking-bubble">
              <span class="dot"></span><span class="dot"></span><span class="dot"></span>
            </div>
          </div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="input-wrapper">
        <div class="input-box">
          <el-input
            v-model="inputQuery"
            placeholder="输入您的问题，按 Enter 发送..."
            type="textarea"
            :autosize="{ minRows: 1, maxRows: 5 }"
            resize="none"
            class="chat-input"
            @keydown.enter.prevent="sendMessage"
          />
          <el-button 
            type="primary" 
            circle 
            class="send-btn"
            :disabled="!inputQuery.trim() || thinking"
            @click="sendMessage"
          >
            <el-icon><Position /></el-icon>
          </el-button>
        </div>
        <div class="input-footer">
          <span>按 Enter 发送，Shift + Enter 换行</span>
        </div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import { 
  UploadFilled, User, Service, Position, Collection, 
  CircleCheckFilled, Delete, Document, Search, ChatLineRound,
  CollectionTag, ArrowDown
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import axios from 'axios'
import MarkdownIt from 'markdown-it'

const md = new MarkdownIt({ html: true, breaks: true, linkify: true })
const API_BASE = 'http://localhost:5000/api/vector'

// State
const collectionName = ref('agent_rag')
const uploadRef = ref(null)
const fileToUpload = ref(null)
const uploading = ref(false)
const uploadSuccess = ref(false)

const inputQuery = ref('')
const messages = ref([])
const thinking = ref(false)
const messagesRef = ref(null)

// Markdown 渲染
const renderMarkdown = (text) => {
  return md.render(text || '')
}

// Toggle Sources
const toggleSources = (index) => {
  const msg = messages.value[index]
  msg.showSources = !msg.showSources
}

// 清空历史
const clearHistory = () => {
  messages.value = []
}

// File Upload Logic
const handleFileChange = (file) => {
  fileToUpload.value = file.raw
  uploadSuccess.value = false
}

const handleExceed = () => {
  ElMessage.warning('每次仅支持上传一个文件，请移除旧文件后再试')
}

const submitUpload = async () => {
  if (!fileToUpload.value) {
    ElMessage.warning('请先选择文件')
    return
  }

  uploading.value = true
  uploadSuccess.value = false

  try {
    const formData = new FormData()
    formData.append('file', fileToUpload.value)
    formData.append('collection_name', collectionName.value)
    
    const response = await axios.post(`${API_BASE}/upload_file`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })

    if (response.data.success) {
      ElMessage.success('上传并入库成功！')
      uploadSuccess.value = true
      uploadRef.value.clearFiles()
      fileToUpload.value = null
    }
  } catch (error) {
    console.error(error)
    ElMessage.error(error.response?.data?.message || '上传失败')
  } finally {
    uploading.value = false
  }
}

// Chat Logic
const sendMessage = async () => {
  const query = inputQuery.value.trim()
  if (!query || thinking.value) return
  
  messages.value.push({
    role: 'user',
    content: query
  })
  
  inputQuery.value = ''
  thinking.value = true
  scrollToBottom()

  try {
    const response = await axios.post(`${API_BASE}/query`, {
      question: query,
      collection_name: collectionName.value
    })

    if (response.data.success) {
      messages.value.push({
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        showSources: false // 默认折叠引用
      })
    } else {
       messages.value.push({
        role: 'assistant',
        content: '抱歉，我遇到了一些问题：' + response.data.message
      })
    }
  } catch (error) {
     messages.value.push({
        role: 'assistant',
        content: '网络错误或服务不可用，请检查后端服务是否启动。'
      })
  } finally {
    thinking.value = false
    scrollToBottom()
  }
}

const scrollToBottom = () => {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}
</script>

<style scoped>
/* 全局布局 */
.app-layout {
  display: flex;
  height: 100vh;
  width: 100vw;
  background-color: #f5f9f5;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  overflow: hidden;
}

/* 侧边栏样式 - 浅绿色 */
.sidebar {
  width: 320px;
  background: #e8f5e9;
  color: #2c3e2f;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 12px rgba(0, 0, 0, 0.05);
  z-index: 10;
}

.logo-area {
  padding: 30px 20px;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid #c8e6c9;
}

.logo-icon {
  font-size: 28px;
}

.logo-area h1 {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
  letter-spacing: 1px;
  color: #2e7d32;
}

.sidebar-content {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
}

.section-title {
  font-size: 12px;
  color: #5d8c5e;
  margin-bottom: 15px;
  text-transform: uppercase;
  letter-spacing: 1px;
  font-weight: 500;
}

.upload-box {
  margin-bottom: 20px;
}

/* 深度定制 Element Plus 上传组件 */
:deep(.el-upload-dragger) {
  background-color: #fff !important;
  border: 1px dashed #b9dfba !important;
  border-radius: 12px !important;
  transition: all 0.3s;
}

:deep(.el-upload-dragger:hover) {
  border-color: #52c41a !important;
  background-color: #f9fff9 !important;
}

.upload-icon {
  font-size: 40px;
  color: #8bc34a;
  margin-bottom: 10px;
}

.upload-text p {
  color: #2c3e2f;
  margin: 0 0 5px 0;
  font-size: 14px;
}

.upload-text span {
  color: #8d9e8e;
  font-size: 12px;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  font-size: 13px;
  color: #2c3e2f;
  margin-bottom: 8px;
  font-weight: 500;
}

:deep(.custom-input .el-input__wrapper) {
  background-color: #fff;
  box-shadow: none;
  border: 1px solid #d9e8da;
  border-radius: 8px;
}

:deep(.custom-input .el-input__inner) {
  color: #2c3e2f;
}

.action-btn {
  width: 100%;
  height: 44px;
  font-size: 15px;
  border-radius: 8px;
  font-weight: 500;
  background-color: #52c41a !important;
  border-color: #52c41a !important;
}

.action-btn:hover {
  background-color: #3e9e12 !important;
  border-color: #3e9e12 !important;
}

.success-tip {
  margin-top: 15px;
  padding: 10px;
  background: #d9f0da;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #2e7d32;
  font-size: 13px;
}

.sidebar-footer {
  padding: 20px;
  text-align: center;
  border-top: 1px solid #c8e6c9;
  color: #8d9e8e;
  font-size: 12px;
}

/* 主区域样式 */
.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #f5f9f5;
  position: relative;
}

.chat-header {
  height: 70px;
  background: #fff;
  border-bottom: 1px solid #e2f0e3;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 30px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
}

.header-info h2 {
  margin: 0 0 5px 0;
  font-size: 18px;
  color: #2c3e2f;
}

.status-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #7d8e7e;
  background: #eff5ef;
  padding: 2px 8px;
  border-radius: 12px;
}

.status-dot {
  width: 6px;
  height: 6px;
  background: #52c41a;
  border-radius: 50%;
}

.messages-container {
  flex: 1;
  padding: 30px;
  overflow-y: auto;
  scroll-behavior: smooth;
}

/* 欢迎屏幕 */
.welcome-screen {
  height: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: #8d9e8e;
}

.welcome-icon {
  font-size: 60px;
  margin-bottom: 20px;
}

.welcome-screen h3 {
  color: #2c3e2f;
  font-size: 24px;
  margin: 0 0 10px 0;
}

.feature-list {
  display: flex;
  gap: 20px;
  margin-top: 40px;
}

.feature-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 20px;
  background: #fff;
  border-radius: 12px;
  width: 100px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s;
}

.feature-item:hover {
  transform: translateY(-5px);
}

.feature-item .el-icon {
  font-size: 24px;
  color: #52c41a;
}

/* 消息气泡 */
.message-row {
  display: flex;
  gap: 15px;
  margin-bottom: 30px;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

.message-row.user {
  flex-direction: row-reverse;
}

.avatar .el-avatar {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.avatar .assistant {
  background: #52c41a;
}

.avatar .user {
  background: #8bc34a;
}

.message-content {
  max-width: 80%;
}

.bubble {
  padding: 15px 20px;
  border-radius: 16px;
  font-size: 15px;
  line-height: 1.7;
  position: relative;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.user .bubble {
  background: #8bc34a;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.assistant .bubble {
  background: #fff;
  color: #2c3e2f;
  border-bottom-left-radius: 4px;
  border: 1px solid #e2f0e3;
}

/* 引用源卡片 */
.sources-card {
  margin-top: 10px;
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid #e2f0e3;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
}

.sources-header {
  padding: 10px 15px;
  background: #f9fff9;
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  color: #2c3e2f;
  transition: background 0.2s;
}

.sources-header:hover {
  background: #eff9ef;
}

.sources-header .arrow {
  margin-left: auto;
  transition: transform 0.3s;
}

.sources-header .arrow.rotated {
  transform: rotate(180deg);
}

.sources-list {
  padding: 0 15px 15px 15px;
}

.source-item {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px dashed #e2f0e3;
}

.source-item:first-child {
  border-top: none;
}

.source-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 12px;
}

.source-meta .index {
  color: #52c41a;
  font-weight: bold;
}

.source-meta .score {
  color: #8d9e8e;
}

.source-text {
  font-size: 13px;
  color: #5d6e5e;
  line-height: 1.5;
}

/* 输入区域 */
.input-wrapper {
  padding: 20px 30px 30px 30px;
  background: transparent;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.input-box {
  width: 100%;
  max-width: 800px;
  position: relative;
  background: #fff;
  border-radius: 24px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
  border: 1px solid #e2f0e3;
  transition: all 0.3s;
}

.input-box:focus-within {
  box-shadow: 0 4px 20px rgba(82, 196, 26, 0.12);
  border-color: #52c41a;
}

:deep(.chat-input .el-textarea__inner) {
  box-shadow: none;
  border: none;
  background: transparent;
  padding: 15px 50px 15px 20px;
  font-size: 15px;
  resize: none;
}

.send-btn {
  position: absolute;
  right: 10px;
  bottom: 10px;
  width: 36px;
  height: 36px;
  background-color: #52c41a !important;
  border-color: #52c41a !important;
}

.send-btn:hover {
  background-color: #3e9e12 !important;
  border-color: #3e9e12 !important;
}

.input-footer {
  margin-top: 10px;
  font-size: 12px;
  color: #8d9e8e;
}

/* 思考动画 */
.thinking-bubble {
  padding: 10px 15px !important;
}

.thinking .dot {
  display: inline-block;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background-color: #8bc34a;
  margin: 0 3px;
  animation: jump 1.4s infinite ease-in-out;
}

.thinking .dot:nth-child(1) { animation-delay: 0s; }
.thinking .dot:nth-child(2) { animation-delay: 0.2s; }
.thinking .dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes jump {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

/* 消息过渡动画 */
.message-fade-enter-active,
.message-fade-leave-active {
  transition: all 0.4s ease;
}

.message-fade-enter-from,
.message-fade-leave-to {
  opacity: 0;
  transform: translateY(20px);
}

/* Markdown 样式适配 */
:deep(.bubble p) {
  margin: 0 0 8px 0;
}

:deep(.bubble p:last-child) {
  margin: 0;
}

:deep(.bubble ul), :deep(.bubble ol) {
  padding-left: 20px;
  margin: 8px 0;
}

:deep(.bubble code) {
  background: rgba(0, 0, 0, 0.05);
  padding: 2px 4px;
  border-radius: 4px;
  font-family: monospace;
}

.user :deep(.bubble code) {
  background: rgba(255, 255, 255, 0.2);
}
</style>