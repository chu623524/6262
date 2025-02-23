import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
from io import BytesIO

# 设置页面样式
st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .title {
            color: #4CAF50;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            color: #555;
            font-size: 18px;
            text-align: center;
        }
        .result-box {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# 设置模型和标准化器的路径
MODEL_URL = 'https://raw.githubusercontent.com/chu623524/6262/main/RF.pkl'
SCALER_URL = 'https://raw.githubusercontent.com/chu623524/6262/main/scaler.pkl'

@st.cache_resource
def load_model():
    model = joblib.load(BytesIO(requests.get(MODEL_URL).content))
    scaler = joblib.load(BytesIO(requests.get(SCALER_URL).content))
    return model, scaler

model, scaler = load_model()

# 定义类别变量选项
文化程度_options = {0: "小学及以下", 1: "初中", 2: "高中/中专", 3: "大专及以上"}
饮酒状态_options = {0: "不饮酒", 1: "偶尔", 2: "经常"}
家庭月收入_options = {0: "0-2000", 1: "2001-5000", 2: "5001-8000", 3: "8000+"}
创伤时恐惧程度_options = {0: "无", 1: "轻度", 2: "中度", 3: "重度"}
吸烟状态_options = {0: "不吸烟", 1: "偶尔", 2: "经常"}
心理负担_options = {0: "没有", 1: "稍有", 2: "中度", 3: "较重", 4: "严重"}

# 页面标题
st.markdown('<p class="title">PTSD 预测系统</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">创伤后6个月 PTSD 预测</p>', unsafe_allow_html=True)

# 获取用户输入
ASDS = st.number_input("ASDS (分)", value=50.0)
文化程度 = st.selectbox("文化程度", options=list(文化程度_options.keys()), format_func=lambda x: 文化程度_options[x])
饮酒状态 = st.selectbox("饮酒状态", options=list(饮酒状态_options.keys()), format_func=lambda x: 饮酒状态_options[x])
舒张压 = st.number_input("舒张压 (mmHg)", value=80.0)
家庭月收入 = st.selectbox("家庭月收入", options=list(家庭月收入_options.keys()), format_func=lambda x: 家庭月收入_options[x])
中性粒细胞绝对值 = st.number_input("中性粒细胞绝对值 (10^9/L)", value=50.0)
氯 = st.number_input("氯 (mmol/L)", value=50.0)
吸烟状态 = st.selectbox("吸烟状态", options=list(吸烟状态_options.keys()), format_func=lambda x: 吸烟状态_options[x])
焦虑评分 = st.number_input("焦虑评分", value=50.0)
AST_ALT = st.number_input("AST/ALT", value=0.5)
A_G = st.number_input("A/G", value=0.5)
血红蛋白 = st.number_input("血红蛋白 (g/dL)", value=12.0)
心理负担 = st.selectbox("心理负担", options=list(心理负担_options.keys()), format_func=lambda x: 心理负担_options[x])
单核细胞绝对值 = st.number_input("单核细胞绝对值 (10^9/L)", value=50.0)
脉搏 = st.number_input("脉搏 (bpm)", value=70.0)
创伤时恐惧程度 = st.selectbox("创伤时恐惧程度", options=list(创伤时恐惧程度_options.keys()), format_func=lambda x: 创伤时恐惧程度_options[x])

# 预测按钮
if st.button("🔍 预测"):
    input_data = np.array([[ASDS, 文化程度, 饮酒状态, 舒张压, 家庭月收入, 中性粒细胞绝对值, 氯, 吸烟状态, 焦虑评分, AST_ALT, A_G, 血红蛋白, 心理负担, 单核细胞绝对值, 脉搏, 创伤时恐惧程度]])
    
    # 标准化
    input_scaled = scaler.transform(input_data)
    
    # 进行预测
    prediction_prob = model.predict_proba(input_scaled)[0, 1]  # PTSD的概率

    # 结果显示优化
    if prediction_prob > 0.5:
        st.markdown(
            f'<div class="result-box" style="background-color: #FF4B4B; color: white;">'
            f'⚠️ 你的 PTSD 发生风险较高（{prediction_prob:.2%}）'
            f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="result-box" style="background-color: #4CAF50; color: white;">'
            f'✅ 你的 PTSD 发生风险较低（{prediction_prob:.2%}）'
            f'</div>', unsafe_allow_html=True)
    
    # 进度条显示风险
    st.progress(prediction_prob)
