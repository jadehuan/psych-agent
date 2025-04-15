import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="心理对话助手", page_icon="🧠")
st.title("🧠 心理对话小助手")
st.write("欢迎倾诉你的烦恼，我会陪着你。")

# 加载情绪分析模型
with st.spinner("加载模型中..."):
    classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 用户输入
user_input = st.text_input("你现在的感受是？", placeholder="例如：我最近压力很大，总是睡不好...")

# 处理输入
if user_input:
    result = classifier(user_input)
    label = result[0]['label']  # 1 star ~ 5 stars 表示情绪等级
    score = float(result[0]['score'])

    st.markdown(f"🔍 模型判断情绪等级为：**{label}**，置信度：{score:.2f}")

    # 简单情绪映射
    if "1" in label or "2" in label:
        st.markdown("💙 听起来你现在很不容易，我会一直在这儿陪你。可以先试试闭上眼睛，深呼吸三次。")
    elif "3" in label:
        st.markdown("🟡 我理解你的感受，说出来就已经很棒了。")
    elif "4" in label or "5" in label:
        st.markdown("🌞 你现在状态挺好，有什么开心的事想说说吗？")

    # 建议模块
    st.markdown("📝 **小建议**：写下一件今天让你稍感轻松或感激的事情，也许会帮助你慢慢好起来。")
