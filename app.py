import streamlit as st
from transformers import pipeline
import random
import time

# --- 1. 頁面設定與 CSS 美化 ---
st.set_page_config(
    page_title="AI 文本偵測實驗室",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
    }
    .result-card {
        padding: 20px;
        border-radius: 15px;
        background-color: #f0f2f6;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 準備範例資料集 ---
AI_EXAMPLES = [
    "Artificial Intelligence refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
    "In conclusion, the impact of climate change is undeniable. We must take immediate action to reduce carbon emissions.",
    "As an AI language model, I do not have personal feelings or opinions. However, I can provide you with information.",
    "To implement a binary search tree in Python, you first need to define a Node class. Each node will contain a value.",
    "The intricate dance of celestial bodies has fascinated humanity for millennia."
]

HUMAN_EXAMPLES = [
    "I literally just spilled coffee all over my laptop... ugh, this is the worst start to a Monday ever.",
    "I think the movie was okay, but honestly, the ending felt kinda rushed. Like, why did they do that?",
    "Hey guys, just checking in. I won't be able to make it to the meeting tmrw, my kid is sick.",
    "OMG you have to try this new pizza place! The crust is so crispy and the cheese is just... wow.",
    "Im not sure if this is the right way to do it, but i usually just wing it and hope for the best."
]

# --- 3. 定義功能函數 (直接修改 Widget 的 Key) ---

# 重點修正 1: 直接操作 'user_input_area' 這個 key
def fill_ai_text():
    st.session_state['user_input_area'] = random.choice(AI_EXAMPLES)

def fill_human_text():
    st.session_state['user_input_area'] = random.choice(HUMAN_EXAMPLES)

def clear_text():
    st.session_state['user_input_area'] = ""

@st.cache_resource
def load_model():
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    return pipeline("text-classification", model=model_name, top_k=None)

# --- 4. 介面佈局 ---

with st.sidebar:
    st.title("關於本工具")
    st.info("此工具利用 RoBERTa 模型來分辨文本是由人類撰寫還是 AI 生成。")
    st.markdown("### 使用指南")
    st.markdown("1. 點擊範例按鈕 或 自行輸入")
    st.markdown("2. 點擊「開始偵測」")

st.title("🧬 AI vs Human 文本鑑識")

# 按鈕區
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn1:
    st.button("🤖 隨機 AI 語氣", on_click=fill_ai_text)
with col_btn2:
    st.button("🧑 隨機人類語氣", on_click=fill_human_text)
with col_btn3:
    st.button("🗑️ 清空內容", on_click=clear_text)

# 重點修正 2: 這裡移除了 'value=' 參數，因為 key 會自動管理值
# 重點修正 3: 確保 session_state 裡有這個 key，避免報錯
if 'user_input_area' not in st.session_state:
    st.session_state['user_input_area'] = ""

txt_input = st.text_area(
    "在此輸入文章內容 (建議英文效果最佳)：",
    height=200,
    key='user_input_area', # 這是唯一的識別碼，直接連動 session state
    placeholder="Waiting for input..."
)

# 載入模型
classifier = load_model()

# 分析按鈕
if st.button("🚀 開始偵測", type="primary"):
    if not txt_input.strip():
        st.warning("⚠️ 請先輸入文字內容！")
    else:
        with st.spinner("🧠 AI 正在分析語法特徵..."):
            time.sleep(0.5) 
            try:
                results = classifier(txt_input, truncation=True, max_length=512)
                scores = {item['label']: item['score'] for item in results[0]}
                ai_score = scores.get('ChatGPT', scores.get('Fake', 0.0))
                human_score = scores.get('Human', scores.get('Real', 0.0))
                
                total = ai_score + human_score
                ai_prob = (ai_score / total) * 100
                human_prob = (human_score / total) * 100
                
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                st.subheader("📊 分析報告")
                
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("🤖 AI 相似度", f"{ai_prob:.1f}%")
                with m_col2:
                    st.metric("🧑 人類相似度", f"{human_prob:.1f}%")

                st.write("")
                if ai_prob > 50:
                    st.progress(int(ai_prob), text="傾向 AI 生成")
                else:
                    st.progress(int(ai_prob), text="傾向人類撰寫")

                st.write("---")
                if ai_prob > 80:
                    st.error("🚨 極高機率是由 AI 生成的。")
                elif ai_prob > 50:
                    st.warning("⚠️ 可能包含 AI 生成的內容。")
                else:
                    st.success("✅ 極高機率是由人類撰寫的。")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"錯誤：{e}")
