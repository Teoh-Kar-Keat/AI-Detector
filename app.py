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

# 自定義 CSS 來美化按鈕和區塊
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
    .big-font {
        font-size: 20px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 準備範例資料集 (模擬資料庫) ---
# 為了展示效果，這裡選用英文，因為該模型對英文最準確
AI_EXAMPLES = [
    "Artificial Intelligence refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    "In conclusion, the impact of climate change is undeniable. We must take immediate action to reduce carbon emissions and promote renewable energy sources. Governments, corporations, and individuals all have a role to play in preserving our planet for future generations.",
    "As an AI language model, I do not have personal feelings or opinions. However, I can provide you with information regarding the topic you are asking about based on the data I have been trained on up until September 2021.",
    "To implement a binary search tree in Python, you first need to define a Node class. Each node will contain a value, a left child, and a right child. Recursion is typically used for insertion and search operations.",
    "The intricate dance of celestial bodies has fascinated humanity for millennia. From the ancient astronomers mapping the stars to modern telescopes peering into the depths of the universe, our quest to understand the cosmos is a testament to human curiosity."
]

HUMAN_EXAMPLES = [
    "I literally just spilled coffee all over my laptop... ugh, this is the worst start to a Monday ever. Does anyone know a good repair shop in downtown? Pls help!",
    "I think the movie was okay, but honestly, the ending felt kinda rushed. Like, why did the main character just leave without saying anything? It didn't make sense to me personally.",
    "Hey guys, just checking in. I won't be able to make it to the meeting tmrw, my kid is sick. I'll catch up on the notes later. Thanks!",
    "OMG you have to try this new pizza place! The crust is so crispy and the cheese is just... wow. 10/10 would recommend.",
    "Im not sure if this is the right way to do it, but i usually just wing it and hope for the best. works 60% of the time, every time lol."
]

# --- 3. 初始化 Session State ---
# 這一步很重要，用來儲存 Text Area 目前的內容
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""

# --- 4. 定義功能函數 ---
@st.cache_resource
def load_model():
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    classifier = pipeline("text-classification", model=model_name, top_k=None)
    return classifier

def fill_ai_text():
    st.session_state['user_input'] = random.choice(AI_EXAMPLES)

def fill_human_text():
    st.session_state['user_input'] = random.choice(HUMAN_EXAMPLES)

def clear_text():
    st.session_state['user_input'] = ""

# --- 5. 介面佈局 ---

# 側邊欄
with st.sidebar:
    st.title("關於本工具")
    st.info("此工具利用 RoBERTa 模型來分辨文本是由人類撰寫還是 AI 生成。")
    st.markdown("### 使用指南")
    st.markdown("1. 輸入文字 或 點擊範例按鈕")
    st.markdown("2. 點擊「開始偵測」")
    st.markdown("3. 查看詳細分析結果")
    st.markdown("---")
    st.caption("Model: Hello-SimpleAI/chatgpt-detector-roberta")

# 主標題
st.title("🧬 AI vs Human 文本鑑識")
st.markdown("### 🕵️ 貼上文章，立即揭穿真偽")

# 按鈕區 (使用 Columns 排版)
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])

with col_btn1:
    st.button("🤖 隨機 AI 語氣", on_click=fill_ai_text, help="自動填入一段 AI 生成風格的文字")
with col_btn2:
    st.button("🧑 隨機人類語氣", on_click=fill_human_text, help="自動填入一段人類口語風格的文字")
with col_btn3:
    st.button("🗑️ 清空內容", on_click=clear_text)

# 文字輸入區 (綁定 Session State)
txt_input = st.text_area(
    "在此輸入文章內容 (建議英文效果最佳)：",
    value=st.session_state['user_input'],
    height=200,
    key='user_input_area',  # 注意：這裡只是一個 key，實際連動要靠下面的邏輯
    placeholder="Waiting for input..."
)

# 讓 text_area 的改變同步回 session_state (為了讓手動輸入也能被記住)
st.session_state['user_input'] = txt_input

# 載入模型
classifier = load_model()

# 分析按鈕
if st.button("🚀 開始偵測", type="primary"):
    if not txt_input.strip():
        st.warning("⚠️ 請先輸入文字內容！")
    else:
        with st.spinner("🧠 AI 正在分析語法特徵..."):
            # 模擬一點延遲感，增加 UX 體驗
            time.sleep(0.5) 
            
            try:
                # 執行預測
                results = classifier(txt_input, truncation=True, max_length=512)
                
                # 處理數據
                scores = {item['label']: item['score'] for item in results[0]}
                ai_score = scores.get('ChatGPT', scores.get('Fake', 0.0))
                human_score = scores.get('Human', scores.get('Real', 0.0))
                
                total = ai_score + human_score
                ai_prob = (ai_score / total) * 100
                human_prob = (human_score / total) * 100
                
                # --- 結果呈現區 ---
                st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                
                st.subheader("📊 分析報告")
                
                # 指標卡片
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("🤖 AI 相似度", f"{ai_prob:.1f}%", delta=f"{ai_prob-50:.1f}%" if ai_prob > 50 else None, delta_color="inverse")
                with m_col2:
                    st.metric("🧑 人類相似度", f"{human_prob:.1f}%", delta=f"{human_prob-50:.1f}%" if human_prob > 50 else None)

                # 進度條
                st.write("") # Spacer
                st.write("判斷傾向：")
                if ai_prob > 50:
                    bar_color = "red"
                    st.progress(int(ai_prob), text="傾向 AI 生成")
                else:
                    bar_color = "green"
                    st.progress(int(ai_prob), text="傾向人類撰寫")

                # 文字結論
                st.write("---")
                if ai_prob > 80:
                    st.error("🚨 **極高風險**：這段文字非常有可能是由 AI 生成的。\n\n特徵：語句結構過於完美、缺乏情感波動或使用了常見的 AI 慣用語。")
                elif ai_prob > 50:
                    st.warning("⚠️ **中度風險**：這段文字包含 AI 生成的特徵，但也可能是經過潤飾的人類文字。")
                else:
                    st.success("✅ **通過驗證**：這段文字看起來很自然，極高機率由人類撰寫。\n\n特徵：包含不規則語法、俚語、強烈的個人語氣或拼寫變化。")
                
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"分析時發生錯誤：{e}")
