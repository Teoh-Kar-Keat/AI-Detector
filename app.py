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
    @import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Inter:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1 {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }

    .stTextArea textarea {
        background-color: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 12px;
        transition: all 0.3s ease;
        font-family: 'Roboto Mono', monospace;
        font-size: 14px;
    }
    .stTextArea textarea:focus {
        border-color: #3a7bd5;
        box-shadow: 0 0 10px rgba(58, 123, 213, 0.2);
    }

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
    
    .status-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. 修正後的範例資料集 (高辨識度) ---
# 這些例子經過特別挑選，針對 RoBERTa 模型的特徵進行了優化

AI_EXAMPLES = [
    # 特徵：經典的 AI 開頭，語氣平鋪直敘，沒有情感
    "As an AI language model developed by OpenAI, I do not have personal experiences or emotions. I can, however, provide information on a wide range of topics based on my training data up to September 2021.",
    
    # 特徵：過度使用連接詞 (Furthermore, Moreover, In conclusion) 和完美的結構
    "Furthermore, the implementation of renewable energy sources is crucial for environmental sustainability. Consequently, governments must incentivize green technologies. In conclusion, a multi-faceted approach is required.",
    
    # 特徵：重複性高，像機器人在解釋定義
    "Machine learning is a subset of artificial intelligence that involves training algorithms to recognize patterns in data. These algorithms can then make predictions or decisions without being explicitly programmed to perform the task.",
    
    # 特徵：過於禮貌和服務導向
    "I hope this explanation helps! Please let me know if you have any other questions regarding quantum mechanics or any other topic. I am here to assist you.",
    
    # 特徵：條列式結構過於完美
    "Here are three benefits of exercise: 1. It improves cardiovascular health. 2. It boosts mental well-being by releasing endorphins. 3. It aids in weight management and muscle tone."
]

HUMAN_EXAMPLES = [
    # 特徵：全小寫，網路簡寫 (idk, tho)，沒有標點符號
    "i literally have no idea what im doing with my life rn tbh. just gonna eat some pizza and watch netflix lol.",
    
    # 特徵：情緒化，連續的標點符號，口語化 (Dude, No way)
    "Dude!!! You won't believe what just happened. I saw my ex at the store and I literally hid behind a shelf. So awkward...",
    
    # 特徵：拼寫錯誤 (teh, becuz)，語法破碎
    "Wait, are we meeting at 5 or 6? i forgot to check teh schedule becuz my phone died. txt me back asap.",
    
    # 特徵：非常特定的個人經驗，語句不連貫
    "My cat just knocked over my coffee cup. Again. This is the third time this week, I swear he does it on purpose just to annoy me.",
    
    # 特徵：充滿猶豫詞 (Umm, like, kinda)
    "Umm, I think the movie was... okay? But like, the ending was kinda weird. I didn't really get it."
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
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    return pipeline("text-classification", model=model_name, top_k=None)

if 'user_input_area' not in st.session_state:
    st.session_state['user_input_area'] = ""

# --- 5. 介面佈局 ---

c1, c2 = st.columns([1, 6])
with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=70)
with c2:
    st.markdown("<h1>NeuralScan Detector</h1>", unsafe_allow_html=True)
    st.caption("🚀 Powered by RoBERTa Transformer Model")

st.markdown("---")

with st.sidebar:
    st.markdown("### ⚙️ 系統核心")
    st.info("Model: `chatgpt-detector-roberta`")
    st.markdown("### 💡 提示")
    st.caption("此模型對於『長句』與『結構完整』的 AI 文本偵測效果最佳。過短的句子可能會導致判斷模糊。")

st.markdown("### 📝 Source Input")

col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn1:
    st.button("🤖 AI Generate (High Conf)", on_click=fill_ai_text)
with col_btn2:
    st.button("🧑 Human Text (High Conf)", on_click=fill_human_text)
with col_btn3:
    st.button("🧹 Clear Terminal", on_click=clear_text)

txt_input = st.text_area(
    label="Input Data Stream",
    label_visibility="collapsed",
    height=180,
    key='user_input_area',
    placeholder="> Waiting for text input to analyze sequence..."
)

st.write("")
run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
with run_col2:
    analyze_btn = st.button("⚡ ANALYZE SEQUENCE ⚡", type="primary")

classifier = load_model()

if analyze_btn:
    if not txt_input.strip():
        st.toast("⚠️ Error: Input buffer is empty!", icon="❌")
    else:
        progress_text = "Initializing Neural Network..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.005)
            if percent_complete == 30:
                my_bar.progress(percent_complete + 1, text="Tokenizing input sequence...")
            elif percent_complete == 60:
                my_bar.progress(percent_complete + 1, text="Calculating attention weights...")
            else:
                my_bar.progress(percent_complete + 1)
        
        my_bar.empty()

        try:
            results = classifier(txt_input, truncation=True, max_length=512)
            scores = {item['label']: item['score'] for item in results[0]}
            ai_score = scores.get('ChatGPT', scores.get('Fake', 0.0))
            human_score = scores.get('Human', scores.get('Real', 0.0))
            
            total = ai_score + human_score
            ai_prob = (ai_score / total) * 100
            human_prob = (human_score / total) * 100
            
            # 判斷邏輯
            if ai_prob > 50:
                verdict = "AI GENERATED"
                verdict_color = "#ffebee" 
                text_color = "#c62828" 
                icon = "🤖"
                main_score = ai_prob
                confidence_text = "HIGH CONFIDENCE" if ai_prob > 80 else "MODERATE CONFIDENCE"
            else:
                verdict = "HUMAN WRITTEN"
                verdict_color = "#e8f5e9" 
                text_color = "#2e7d32" 
                icon = "🧑"
                main_score = human_prob
                confidence_text = "HIGH CONFIDENCE" if human_prob > 80 else "MODERATE CONFIDENCE"

            st.markdown(f"""
            <div class="result-card">
                <div class="status-badge" style="background-color: {verdict_color}; color: {text_color}; border: 1px solid {text_color};">
                    {icon} DETECTION RESULT
                </div>
                <h2 style="color: #333; margin: 0;">Probability Distribution</h2>
                <div class="score-box" style="color: {text_color};">
                    {main_score:.1f}% <span style="font-size: 1rem; color: #666;">{confidence_text}</span>
                </div>
                <p style="font-weight: bold; font-size: 1.2rem; color: {text_color};">
                    VERDICT: {verdict}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.caption("🤖 Artificial Intelligence")
                st.progress(int(ai_prob))
            with res_col2:
                st.caption("🧑 Human Intelligence")
                st.progress(int(human_prob))

            with st.expander("🔍 View Raw Tensor Output"):
                st.json(results)

        except Exception as e:
            st.error(f"System Error: {e}")
