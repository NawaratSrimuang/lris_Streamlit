import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.datasets import load_iris
 
 
# ==================== ตั้งค่าหน้าเว็บ ====================
st.set_page_config(
    page_title="🌷 Iris Garden Classifier",
    page_icon="🌷",
    layout="wide"
)
 
 
# ==================== CSS ธีมน่ารัก ====================
st.markdown("""
<style>
 
.stApp {
    background: linear-gradient(135deg, #fff5fb, #f3f0ff);
}
 
/* ชื่อเว็บ */
.main-title {
    color: #8e5aa8;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}
 
/* คำอธิบาย */
.subtitle {
    color: #9b7aaa;
    font-size: 17px;
    margin-bottom: 20px;
}
 
/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fce4f3, #eee5ff);
}
 
/* Prediction Card */
.cute-card {
    background: linear-gradient(135deg, #f8cce5, #d9c2ff);
    padding: 30px;
    border-radius: 25px;
    text-align: center;
    color: #5c426c;
    box-shadow: 0 6px 20px rgba(150, 100, 170, 0.15);
    margin-bottom: 20px;
    border: 2px solid white;
}
 
.cute-card h1 {
    font-size: 40px;
    margin: 10px 0;
    color: #6d477d;
}
 
.cute-card h3 {
    margin: 0;
    color: #80548f;
}
 
/* ปุ่ม */
.stButton > button {
    background: linear-gradient(90deg, #d99ac5, #a995d6);
    color: white;
    border: none;
    border-radius: 20px;
    font-weight: bold;
    padding: 10px;
}
 
.stButton > button:hover {
    background: linear-gradient(90deg, #c985b4, #9583c8);
    color: white;
}
 
</style>
""", unsafe_allow_html=True)
 
 
# ==================== โหลด Iris Dataset ====================
iris = load_iris()
 
feature_names = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width"
]
 
species_names = iris.target_names
 
 
# ==================== คำนวณค่าเฉลี่ย Dataset ====================
df_iris = pd.DataFrame(
    iris.data,
    columns=feature_names
)
 
dataset_averages = df_iris.mean().values
 
 
# ==================== โหลดโมเดล ====================
@st.cache_resource
def load_model():
 
    try:
        model = joblib.load("iris_model.pkl")
 
    except Exception:
 
        from sklearn.ensemble import RandomForestClassifier
 
        model = RandomForestClassifier(
            random_state=42
        )
 
        model.fit(
            iris.data,
            iris.target
        )
 
    return model
 
 
model = load_model()
 
 
# ==================== SIDEBAR ====================
with st.sidebar:
 
    st.markdown("## 🌷 Iris Garden")
 
    st.caption(
        "ปรับค่าของดอกไม้ที่ต้องการจำแนกได้เลย 💕"
    )
 
    st.markdown("---")
 
    sepal_length = st.slider(
        "🌸 Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.80,
        step=0.01
    )
 
    sepal_width = st.slider(
        "🍃 Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.00,
        step=0.01
    )
 
    petal_length = st.slider(
        "🌷 Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.00,
        step=0.01
    )
 
    petal_width = st.slider(
        "🌼 Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.20,
        step=0.01
    )
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    predict_btn = st.button(
        "🌷 Predict Species 💕",
        use_container_width=True,
        type="primary"
    )
 
 
# ==================== MAIN CONTENT ====================
 
st.markdown(
    '<div class="main-title">🌷 Iris Garden Classifier</div>',
    unsafe_allow_html=True
)
 
st.markdown(
    '<div class="subtitle">'
    '🌸 ระบบจำแนกสายพันธุ์ดอกไอริสด้วย Machine Learning 🌸'
    '</div>',
    unsafe_allow_html=True
)
 
st.markdown("---")
 
 
# ==================== Prediction ====================
 
input_data = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])
 
prediction = model.predict(input_data)[0]
 
probabilities = model.predict_proba(input_data)[0]
 
predicted_species = species_names[prediction].capitalize()
 
confidence = probabilities[prediction] * 100
 
 
# ==================== แบ่งหน้าจอ ====================
 
col1, col2 = st.columns([1.1, 1])
 
 
# =========================================================
# COLUMN 1 : INPUT VISUALIZATION
# =========================================================
 
with col1:
 
    st.subheader("📊 Input Visualization")
 
    st.caption(
        "เปรียบเทียบค่าที่ป้อนกับค่าเฉลี่ยของ Dataset 🌱"
    )
 
    fig_bar = go.Figure()
 
 
    # ค่าที่ผู้ใช้ป้อน
    fig_bar.add_trace(
        go.Bar(
            x=feature_names,
            y=[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ],
            name="🌸 Your Input",
            marker_color="#E8A8C8",
            text=[
                f"{sepal_length:.1f}",
                f"{sepal_width:.1f}",
                f"{petal_length:.1f}",
                f"{petal_width:.1f}"
            ],
            textposition="auto"
        )
    )
 
 
    # ค่าเฉลี่ย Dataset
    fig_bar.add_trace(
        go.Bar(
            x=feature_names,
            y=dataset_averages,
            name="🌿 Dataset Average",
            marker_color="#AFA1D9",
            text=[
                f"{v:.2f}"
                for v in dataset_averages
            ],
            textposition="auto"
        )
    )
 
 
    fig_bar.update_layout(
        barmode="group",
        height=380,
        margin=dict(
            l=20,
            r=20,
            t=20,
            b=40
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="🌷 Features",
        yaxis_title="Value (cm)"
    )
 
    st.plotly_chart(
        fig_bar,
        use_container_width=True
    )
 
 
# =========================================================
# COLUMN 2 : PREDICTION RESULT
# =========================================================
 
with col2:
 
    st.subheader("🔮 Prediction Result")
 
    st.markdown(
        f"""
        <div class="cute-card">
 
            <div style="font-size:45px;">
                🌷 ✨ 🌸
            </div>
 
            <h3>
                Predicted Species
            </h3>
 
            <h1>
                {predicted_species}
            </h1>
 
            <p style="font-size:18px;">
                💕 Confidence:
                <b>{confidence:.1f}%</b>
            </p>
 
            <div style="
                background:#ffffff;
                border-radius:20px;
                padding:8px;
                margin-top:15px;
            ">
 
                <div style="
                    width:{confidence:.1f}%;
                    background:linear-gradient(
                        90deg,
                        #e7a8c8,
                        #a995d6
                    );
                    height:12px;
                    border-radius:20px;
                ">
                </div>
 
            </div>
 
        </div>
        """,
        unsafe_allow_html=True
    )
 
 
    # ==================== Probability ====================
 
    st.markdown(
        "##### 🌈 Probability Distribution"
    )
 
    colors = [
        "#F3B6D2",
        "#BFADE3",
        "#A8D8D8"
    ]
 
    # ทำให้สายพันธุ์ที่ถูกทำนายเด่นขึ้น
    colors[prediction] = "#D982B5"
 
 
    fig_prob = go.Figure()
 
 
    fig_prob.add_trace(
        go.Bar(
            x=[
                s.capitalize()
                for s in species_names
            ],
            y=probabilities * 100,
            marker_color=colors,
            text=[
                f"{p*100:.1f}%"
                for p in probabilities
            ],
            textposition="outside",
            marker_line=dict(
                width=1,
                color="#FFFFFF"
            )
        )
    )
 
 
    fig_prob.update_layout(
        height=250,
        margin=dict(
            l=20,
            r=20,
            t=10,
            b=40
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.55)",
        yaxis=dict(
            title="Probability (%)",
            range=[0, 115]
        ),
        xaxis_title="🌷 Species"
    )
 
 
    st.plotly_chart(
        fig_prob,
        use_container_width=True
    )
 
 
# ==================== FOOTER ====================
 
st.markdown("---")
 
st.markdown(
    """
    <div style="
        text-align:center;
        color:#9B7AAA;
        font-size:14px;
        padding:10px;
    ">
 
        🌸 Welcome to Iris Garden Classifier 🌸<br>
 
        💕 Machine Learning Flower Classifier 💕
 
    </div>
    """,
    unsafe_allow_html=True
)
 