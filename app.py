import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
from sklearn.datasets import load_iris


# =========================================================
# ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="Iris Garden Classifier",
    page_icon="🌷",
    layout="wide"
)


# =========================================================
# CSS ตกแต่งเว็บ
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #fff7fc, #f4f0ff);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #fde8f4,
        #eee7ff
    );
}

/* ปุ่ม */
.stButton > button {
    background: linear-gradient(
        90deg,
        #d99ac5,
        #a995d6
    );
    color: white;
    border: none;
    border-radius: 20px;
    font-weight: bold;
    padding: 10px;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #c985b4,
        #9583c8
    );
    color: white;
}

/* หัวข้อ */
h1 {
    color: #8e5aa8;
}

h2, h3 {
    color: #80548f;
}

/* Progress bar */
div[data-testid="stProgress"] > div > div {
    background-color: #d99ac5;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# โหลด Iris Dataset
# =========================================================

iris = load_iris()

feature_names = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width"
]

species_names = iris.target_names


# =========================================================
# คำนวณค่าเฉลี่ยของ Dataset
# =========================================================

df_iris = pd.DataFrame(
    iris.data,
    columns=feature_names
)

dataset_averages = df_iris.mean().values


# =========================================================
# โหลดโมเดล
# =========================================================

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


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🌷 Iris Garden")

    st.caption(
        "ปรับค่าของดอกไม้ที่ต้องการจำแนกได้เลย 💕"
    )

    st.markdown("---")

    # Sepal Length
    sepal_length = st.slider(
        "🌸 Sepal Length (cm)",
        min_value=4.0,
        max_value=8.0,
        value=5.80,
        step=0.01
    )

    # Sepal Width
    sepal_width = st.slider(
        "🍃 Sepal Width (cm)",
        min_value=2.0,
        max_value=4.5,
        value=3.00,
        step=0.01
    )

    # Petal Length
    petal_length = st.slider(
        "🌷 Petal Length (cm)",
        min_value=1.0,
        max_value=7.0,
        value=4.00,
        step=0.01
    )

    # Petal Width
    petal_width = st.slider(
        "🌼 Petal Width (cm)",
        min_value=0.1,
        max_value=2.5,
        value=1.20,
        step=0.01
    )

    st.markdown("")

    predict_btn = st.button(
        "🌷 Predict Species 💕",
        use_container_width=True,
        type="primary"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🌷 Iris Garden Classifier")

st.write(
    "🌸 ระบบจำแนกสายพันธุ์ดอกไอริสด้วย Machine Learning 🌸"
)

st.divider()


# =========================================================
# เตรียมข้อมูลสำหรับการทำนาย
# =========================================================

input_data = np.array([[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]])


# =========================================================
# ทำนาย
# =========================================================

prediction = model.predict(input_data)[0]

probabilities = model.predict_proba(input_data)[0]

predicted_species = species_names[prediction].capitalize()

confidence = probabilities[prediction] * 100


# =========================================================
# แบ่งหน้าจอเป็น 2 คอลัมน์
# =========================================================

col1, col2 = st.columns([1.1, 1])


# =========================================================
# COLUMN 1
# INPUT VISUALIZATION
# =========================================================

with col1:

    st.subheader("📊 Input Visualization")

    st.caption(
        "เปรียบเทียบค่าที่ป้อนกับค่าเฉลี่ยของ Dataset 🌱"
    )

    # สร้างกราฟ
    fig_bar = go.Figure()


    # -----------------------------------------------------
    # ค่าที่ผู้ใช้ป้อน
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # ค่าเฉลี่ย Dataset
    # -----------------------------------------------------

    fig_bar.add_trace(
        go.Bar(
            x=feature_names,

            y=dataset_averages,

            name="🌿 Dataset Average",

            marker_color="#AFA1D9",

            text=[
                f"{value:.2f}"
                for value in dataset_averages
            ],

            textposition="auto"
        )
    )


    # -----------------------------------------------------
    # ตั้งค่ากราฟ
    # -----------------------------------------------------

    fig_bar.update_layout(

        barmode="group",

        height=380,

        margin=dict(
            l=20,
            r=20,
            t=30,
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
# COLUMN 2
# PREDICTION RESULT
# =========================================================

with col2:

    st.subheader("🔮 Prediction Result")

    st.write("")


    # -----------------------------------------------------
    # กล่องผลลัพธ์แบบ Streamlit
    # ไม่ใช้ HTML เพื่อป้องกัน <div> โผล่เป็นข้อความ
    # -----------------------------------------------------

    st.info("🌷 ✨ 🌸")

    st.markdown(
        "### 🌸 Predicted Species"
    )

    st.markdown(
        f"# {predicted_species}"
    )

    st.markdown(
        f"💕 **Confidence: {confidence:.1f}%**"
    )


    # -----------------------------------------------------
    # Confidence Progress
    # -----------------------------------------------------

    st.progress(
        int(confidence),
        text=f"🌸 Confidence {confidence:.1f}%"
    )


    st.write("")


    # -----------------------------------------------------
    # Probability Distribution
    # -----------------------------------------------------

    st.markdown(
        "### 🌈 Probability Distribution"
    )


    # สีของแต่ละสายพันธุ์
    colors = [
        "#F3B6D2",
        "#BFADE3",
        "#A8D8D8"
    ]


    # ทำให้ตัวที่ทำนายเด่นที่สุด
    colors[prediction] = "#D982B5"


    # สร้างกราฟ Probability
    fig_prob = go.Figure()


    fig_prob.add_trace(
        go.Bar(

            x=[
                species.capitalize()
                for species in species_names
            ],

            y=probabilities * 100,

            marker_color=colors,

            text=[
                f"{probability * 100:.1f}%"
                for probability in probabilities
            ],

            textposition="outside",

            marker_line=dict(
                width=1,
                color="#FFFFFF"
            )
        )
    )


    # -----------------------------------------------------
    # ตั้งค่ากราฟ Probability
    # -----------------------------------------------------

    fig_prob.update_layout(

        height=280,

        margin=dict(
            l=20,
            r=20,
            t=20,
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


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    "🌸 **Iris Garden Classifier** 🌸"
)

st.caption(
    "💕 Machine Learning Flower Classifier 💕"
)
