import streamlit as st
from transformers import pipeline
import random
import time

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="NeuralScan | AI 文本偵測",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 2. CSS 極致美化 (AI/科技風格) ---
st.markdown("""
    <style>
    /* 全局字體設定 */
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 標題樣式 */
    h1 {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    /* 文本輸入框美化 */
    .stTextArea textarea {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-family: 'Roboto Mono', monospace; /* 代碼感 */
        font-size: 14px;
    }
    .stTextArea textarea:focus {
        border-color: #3a7bd5;
        box-shadow: 0 0 10px rgba(58, 123, 213, 0.2);
    }

    /* 按鈕美化 */
    div.stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3.2em;
        font-weight: 600;
        border: none;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box_shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* AI 按鈕特定樣式 (Streamlit 無法直接選特定按鈕，這裡做通用優化) */
    
    /* 結果卡片 - 玻璃擬態風格 */
    .result-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 20px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        text-align: center;
    }
    
    .score-box {
        font-family: 'Roboto Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .ai-color { color: #ff4b4b; }
    .human-color { color: #00cc96; }
    
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* 分隔線 */
    hr {
        margin: 2em 0;
        border: 0;
        border-top: 1px solid #eee;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 準備範例資料集 ---
AI_EXAMPLES = [
    "Artificial Intelligence allows machines to model, and even improve upon, the capabilities of the human mind. From the development of self-driving cars to the generation of generative art, AI is reshaping our world.",
    "To summarize, the integration of renewable energy systems is pivotal for sustainable development. Policy frameworks must adapt to facilitate this transition efficiently.",
    "As a large language model trained by OpenAI, I cannot browse the live internet or access personal emails. My purpose is to assist with information processing.",
    "In Python, a decorator is a design pattern that allows you to modify the functionality of a function by wrapping it in another function.",
    "The concept of the metaverse represents a convergence of physical and digital realities, creating a persistent, shared virtual space."
]

HUMAN_EXAMPLES = [
    "Dude, I just saw the craziest thing outside my window. There was this squirrel fighting a pigeon over a bagel lol.",
    "I'm so done with this week. Can we just skip to Friday? I need a nap and a pizza, specifically in that order.",
    "Actually, I think the second season was better than the first. The character development for Sarah was way more realistic.",
    "Has anyone seen my keys? I swear I left them on the counter. This happens every single morning!",
    "wanna grab lunch later? i found this new burger spot nearby looks pretty good."
]

# --- 4. 核心邏輯 ---

def fill_ai_text():
    st.session_state['user_input_area'] = random.choice(AI_EXAMPLES)

def fill_human_text():
    st.session_state['user_input_area'] = random.choice(HUMAN_EXAMPLES)

def clear_text():
    st.session_state['user_input_area'] = ""

@st.cache_resource
def load_model():
    # 使用較輕量的模型
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    return pipeline("text-classification", model=model_name, top_k=None)

# 確保 Session State 存在
if 'user_input_area' not in st.session_state:
    st.session_state['user_input_area'] = ""

# --- 5. 介面佈局 ---

# Header 區域
c1, c2 = st.columns([1, 6])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=70) # 簡單的 AI Icon
with c2:
    st.markdown("<h1>NeuralScan Detector</h1>", unsafe_allow_html=True)
    st.caption("🚀 Powered by RoBERTa Transformer Model")

st.markdown("---")

# 側邊欄
with st.sidebar:
    st.markdown("### ⚙️ 系統核心")
    st.info("Model: `chatgpt-detector-roberta`\n\nBackend: `PyTorch`")
    st.markdown("### 📖 使用指南")
    st.text("1. 輸入或選取範本")
    st.text("2. 執行神經網絡分析")
    st.text("3. 檢視機率分佈")
    st.markdown("---")
    st.caption("Designed for AI research")

# 功能區塊
st.markdown("### 📝 Source Input")

# 功能按鈕列
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn1:
    st.button("🤖 Generate AI Text", on_click=fill_ai_text, help="插入 AI 生成樣本")
with col_btn2:
    st.button("🧑 Generate Human Text", on_click=fill_human_text, help="插入人類撰寫樣本")
with col_btn3:
    st.button("🧹 Clear Terminal", on_click=clear_text)

# 輸入框
txt_input = st.text_area(
    label="Input Data Stream",
    label_visibility="collapsed",
    height=180,
    key='user_input_area',
    placeholder="> Waiting for text input to analyze sequence..."
)

# 執行區塊
st.write("") # Spacer
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    analyze_btn = st.button("⚡ ANALYZE SEQUENCE ⚡", type="primary")

# 模型載入
classifier = load_model()

# 分析邏輯
if analyze_btn:
    if not txt_input.strip():
        st.toast("⚠️ Error: Input buffer is empty!", icon="❌")
    else:
        # 自定義進度條動畫
        progress_text = "Initializing Neural Network..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.005) # 假裝很忙的特效
            if percent_complete == 30:
                my_bar.progress(percent_complete + 1, text="Tokenizing input sequence...")
            elif percent_complete == 60:
                my_bar.progress(percent_complete + 1, text="Calculating attention weights...")
            else:
                my_bar.progress(percent_complete + 1)
        
        my_bar.empty()

        try:
            # 實際預測
            results = classifier(txt_input, truncation=True, max_length=512)
            scores = {item['label']: item['score'] for item in results[0]}
            ai_score = scores.get('ChatGPT', scores.get('Fake', 0.0))
            human_score = scores.get('Human', scores.get('Real', 0.0))
            
            total = ai_score + human_score
            ai_prob = (ai_score / total) * 100
            human_prob = (human_score / total) * 100
            
            # --- 結果顯示區 (HTML/CSS 組裝) ---
            
            # 決定顏色與標籤
            if ai_prob > 50:
                verdict = "AI GENERATED"
                verdict_color = "#ffebee" # 淺紅背景
                text_color = "#c62828" # 深紅文字
                icon = "🤖"
                main_score = ai_prob
            else:
                verdict = "HUMAN WRITTEN"
                verdict_color = "#e8f5e9" # 淺綠背景
                text_color = "#2e7d32" # 深綠文字
                icon = "🧑"
                main_score = human_prob

            st.markdown(f"""
            <div class="result-card">
                <div class="status-badge" style="background-color: {verdict_color}; color: {text_color}; border: 1px solid {text_color};">
                    {icon} DETECTION RESULT
                </div>
                <h2 style="color: #333; margin: 0;">Probability Distribution</h2>
                <div class="score-box" style="color: {text_color};">
                    {main_score:.1f}% <span style="font-size: 1rem; color: #666;">CONFIDENCE</span>
                </div>
                <p style="font-weight: bold; font-size: 1.2rem; color: {text_color};">
                    VERDICT: {verdict}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # 詳細數據與圖表
            st.write("")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.caption("🤖 Artificial Intelligence")
                st.progress(int(ai_prob))
            with res_col2:
                st.caption("🧑 Human Intelligence")
                st.progress(int(human_prob))

            # 技術細節
            with st.expander("🔍 View Raw Tensor Output"):
                st.json(results)
                st.code(f"Input Tokens: {len(txt_input.split())} words\nProcessed Length: {min(len(txt_input), 512)} chars", language="bash")

        except Exception as e:
            st.error(f"System Error: {e}")
