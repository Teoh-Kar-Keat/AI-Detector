import streamlit as st
from transformers import pipeline
import torch

# 設定頁面配置
st.set_page_config(
    page_title="AI vs Human 文本偵測器",
    page_icon="🤖",
    layout="centered"
)

# 快取模型，避免每次重新載入 (Streamlit Cache)
@st.cache_resource
def load_model():
    # 使用 Hugging Face 上較輕量且熱門的 ChatGPT 偵測模型
    model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
    
    # 建立分類 pipeline
    # return_all_scores=True 會同時回傳 Human 和 AI 的機率
    classifier = pipeline("text-classification", model=model_name, top_k=None)
    return classifier

# 側邊欄資訊
st.sidebar.title("關於工具")
st.sidebar.info(
    "此工具使用 Transformer 模型 (RoBERTa) "
    "來分析文本的語法與統計特徵，判斷是否由 AI 生成。"
)
st.sidebar.warning(
    "⚠️ 注意：AI 偵測器並非 100% 準確，"
    "結果僅供參考，請勿作為單一評判標準。"
)

# 主標題
st.title("🤖 AI / 🧑 Human 文章偵測器")
st.markdown("輸入一段文本，AI 將分析其由人類或人工智慧撰寫的可能性。")

# 載入模型 (顯示載入中的 spinner)
with st.spinner("正在載入 AI 偵測模型..."):
    classifier = load_model()

# 文本輸入區
user_input = st.text_area("請在此貼上文章內容 (建議英文效果較佳，中文亦可嘗試)：", height=200)

if st.button("開始分析"):
    if not user_input.strip():
        st.error("請輸入文字內容！")
    else:
        # 進行預測
        # Truncation=True 確保超過 512 tokens 的長文不會報錯
        try:
            results = classifier(user_input, truncation=True, max_length=512)
            
            # 解析結果 (結果通常是一個 list 包含 dict)
            # Hello-SimpleAI 模型的標籤通常是 'Human' 和 'ChatGPT'
            # 我們需要將其標準化
            scores = {item['label']: item['score'] for item in results[0]}
            
            # 取得各別分數 (處理標籤名稱可能不同的情況)
            ai_score = scores.get('ChatGPT', scores.get('Fake', 0.0))
            human_score = scores.get('Human', scores.get('Real', 0.0))
            
            # 確保總和為 1 (雖然 softmax 已經做過，但保險起見)
            total = ai_score + human_score
            ai_prob = (ai_score / total) * 100
            human_prob = (human_score / total) * 100

            # --- 顯示結果 ---
            st.markdown("---")
            st.subheader("📊 分析結果")

            # 使用 Streamlit 的 columns 進行排版
            col1, col2 = st.columns(2)

            with col1:
                st.metric(label="🤖 AI 可能性", value=f"{ai_prob:.1f}%")
            with col2:
                st.metric(label="🧑 人類可能性", value=f"{human_prob:.1f}%")

            # 進度條視覺化
            st.write("AI 傾向程度：")
            st.progress(int(ai_prob))
            
            # 判斷結論
            if ai_prob > 80:
                st.error("🕵️ 結論：這篇文章 **極高機率** 是由 AI 生成的。")
            elif ai_prob > 50:
                st.warning("🤔 結論：這篇文章 **可能** 包含 AI 生成的內容。")
            else:
                st.success("📝 結論：這篇文章 **極高機率** 是由人類撰寫的。")
            
            # 顯示原始數據 (Debug 用，可選)
            with st.expander("查看原始模型數據"):
                st.json(results)

        except Exception as e:
            st.error(f"發生錯誤：{str(e)}")
