# AI-Detector

本專案旨在建構一個高效且具備現代化介面的「AI vs Human 文章偵測器」。在模型選擇上，我們採用了 Hello-SimpleAI/chatgpt-detector-roberta。選擇該模型的主要原因在於，傳統的機器學習方法（如 TF-IDF 搭配邏輯回歸）已難以辨識現代大型語言模型生成的複雜文本，而此模型基於強大的 RoBERTa 架構並針對 ChatGPT 的輸出特徵進行過微調。它能在保持輕量化運算的同時，提供極高的偵測準確率，且運算需求低，非常適合直接部署於 Streamlit Cloud 等無 GPU 的輕量級雲端環境。


在實作方面，我們使用 Python 的 Streamlit 框架快速搭建網頁介面。技術核心是透過 Hugging Face 的 transformers 套件建立自動化推論管線（Pipeline），將使用者輸入的文本轉換為向量進行分析，即時計算出 AI 與人類的機率分佈。為了優化使用者體驗，我們利用 st.session_state 技術解決了元件狀態同步問題，設計了能自動填入「高信心度樣本」的互動按鈕，透過區分充滿拼寫錯誤的人類口語與結構僵化的 AI 慣用語，來驗證模型的準確性。最後，結合自定義 CSS 注入玻璃擬態（Glassmorphism）與霓虹漸層元素，成功將一個單純的分析工具轉化為具備極致科技感的視覺化應用。

https://github.com/user-attachments/assets/8e71a708-4f10-41dd-b4ea-315eeca17441

