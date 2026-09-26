import os
import streamlit as st
import requests
import pandas as pd
from sklearn.datasets import load_breast_cancer


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HEXAMIND",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CONFIG
# =========================================================

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


FEATURE_NAMES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]


# =========================================================
# SESSION STATE
# =========================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    /* REMOVE STREAMLIT TOP HEADER / WHITE GAP */

    header[data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }

    div[data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(0, 170, 255, 0.13),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 80%,
                rgba(110, 60, 255, 0.13),
                transparent 30%
            ),
            linear-gradient(
                135deg,
                #020817 0%,
                #061426 45%,
                #050713 100%
            );

        color: #eaf4ff;
    }


    .main .block-container {
        max-width: 1350px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }


    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #030b18,
                #050a15
            );

        border-right: 1px solid rgba(
            50,
            180,
            255,
            0.20
        );
    }


    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #bfd0e5 !important;
    }


    /* HEXAMIND LOGO */

    .hex-logo {
        width: 88px;
        height: 88px;

        margin: 5px auto 12px auto;

        clip-path: polygon(
            50% 0%,
            93% 25%,
            93% 75%,
            50% 100%,
            7% 75%,
            7% 25%
        );

        background:
            linear-gradient(
                135deg,
                #6f52ff,
                #16cfff
            );

        display: flex;
        align-items: center;
        justify-content: center;

        color: white;

        font-size: 46px;
        font-weight: 900;

        box-shadow:
            0 0 30px rgba(
                0,
                200,
                255,
                0.35
            );
    }


    /* BRAND */

    .brand-name {
        text-align: center;

        font-size: 54px;

        font-weight: 900;

        letter-spacing: 4px;

        background: linear-gradient(
            90deg,
            #ffffff,
            #5edcff,
            #8974ff
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }


    .brand-tagline {
        text-align: center;

        color: #b7d3ef !important;

        font-size: 18px;

        margin-top: -4px;
    }


    .brand-description {
        text-align: center;

        color: #829bb7 !important;

        font-size: 14px;

        margin-top: 5px;

        margin-bottom: 25px;
    }


    /* SIDEBAR */

    .sidebar-brand {
        text-align: center;

        color: #5fdcff !important;

        font-size: 26px;

        font-weight: 900;

        letter-spacing: 2px;

        margin-bottom: 20px;
    }


    .online-box {
        padding: 13px;

        border-radius: 12px;

        background: rgba(
            0,
            180,
            140,
            0.14
        );

        border: 1px solid rgba(
            0,
            230,
            180,
            0.28
        );

        color: #55efbe !important;

        font-weight: 700;
    }


    /* TABS */

    button[data-baseweb="tab"] {
        color: #91a9c4 !important;

        font-size: 16px;

        font-weight: 700;
    }


    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
    }


    /* FEATURE BOXES */

    .feature-box {
        text-align: center;

        padding: 18px 10px;

        border-radius: 14px;

        background: rgba(
            8,
            25,
            48,
            0.72
        );

        border: 1px solid rgba(
            50,
            190,
            255,
            0.18
        );

        min-height: 120px;
    }


    .feature-icon {
        font-size: 35px;

        font-weight: 900;

        margin-bottom: 5px;

        color: #62dcff;
    }


    .feature-title {
        color: #ffffff !important;

        font-weight: 750;

        font-size: 15px;
    }


    .feature-description {
        color: #829bb8 !important;

        font-size: 12px;

        margin-top: 3px;
    }


    /* METRICS */

    div[data-testid="stMetric"] {
        background: rgba(
            8,
            25,
            48,
            0.85
        );

        border: 1px solid rgba(
            50,
            190,
            255,
            0.25
        );

        border-radius: 14px;

        padding: 18px;
    }


    div[data-testid="stMetricLabel"] {
        color: #91a8c2 !important;
    }


    div[data-testid="stMetricValue"] {
        color: #62dcff !important;
    }


    /* BUTTON */

    .stButton > button {
        border-radius: 11px;

        border: 1px solid rgba(
            50,
            195,
            255,
            0.55
        );

        background:
            linear-gradient(
                90deg,
                rgba(0, 160, 255, 0.25),
                rgba(100, 70, 255, 0.28)
            );

        color: #ffffff !important;

        font-weight: 800;

        min-height: 44px;
    }


    .stButton > button:hover {
        border-color: #62dcff;

        box-shadow:
            0 0 20px rgba(
                0,
                190,
                255,
                0.30
            );
    }


    /* CHAT */

    [data-testid="stChatMessage"] {
        background: rgba(
            9,
            25,
            48,
            0.82
        );

        border: 1px solid rgba(
            70,
            180,
            255,
            0.15
        );

        border-radius: 14px;

        margin-bottom: 12px;
    }


    [data-testid="stChatMessage"] p {
        color: #e6f0fc !important;

        font-size: 16px;

        line-height: 1.6;
    }


    [data-testid="stChatInput"] textarea {
        background: #07182d !important;

        color: #eaf4ff !important;

        border: 1px solid #16c8ff !important;

        border-radius: 12px;
    }


    [data-testid="stChatInput"] textarea::placeholder {
        color: #7791ad !important;
    }


    /* FOOTER */

    .hex-footer {
        text-align: center;

        color: #607894 !important;

        font-size: 13px;

        padding-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="hex-logo">6</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand-name">HEXAMIND</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand-tagline">'
    '6 Core Features • 1 Intelligent System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand-description">'
    'Hybrid Quantum-Classical Machine Learning Platform'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">⚛️ HEXAMIND</div>',
        unsafe_allow_html=True
    )

    st.markdown("### ⚙️ System")

    try:

        response = requests.get(
            f"{API_URL}/",
            timeout=5
        )

        if response.status_code == 200:

            st.markdown(
                '<div class="online-box">'
                '● AI Online'
                '</div>',
                unsafe_allow_html=True
            )

        else:

            st.error("AI Offline")

    except requests.exceptions.RequestException:

        st.error("AI Offline")


    st.divider()

    st.markdown("### 🧬 Architecture")

    st.write("⚛️ 4 Qubits")
    st.write("🧬 30 Medical Inputs")
    st.write("🔬 Hybrid QML Model")
    st.write("🧠 AI")


    st.divider()

    st.markdown("### 🔗 Pipeline")

    st.code(
        "User\n"
        " ↓\n"
        "HEXAMIND\n"
        " ↓\n"
        "Hybrid QML\n"
        " ↓\n"
        "AI"
    )


    st.divider()

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#7189a5;
            font-size:13px;
        ">
            HEXAMIND TEAM<br><br>
            6 Core Features • 1 Vision
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TABS
# =========================================================

prediction_tab, chat_tab = st.tabs(
    [
        "🔬 QML Prediction",
        "💬 AI Assistant"
    ]
)


# =========================================================
# PREDICTION TAB
# =========================================================

with prediction_tab:

    # -----------------------------------------------------
    # INTRODUCTION
    # -----------------------------------------------------

    st.header("⚛️ Hybrid Quantum ML Prediction")

    st.write(
        "Use the trained Hybrid Quantum-Classical model "
        "to analyze the provided medical feature input."
    )


    st.divider()


    # -----------------------------------------------------
    # FEATURE SECTION
    # -----------------------------------------------------

    f1, f2, f3, f4 = st.columns(4)


    with f1:

        st.markdown("### ⚛️ Quantum")

        st.caption("Computing")


    with f2:

        st.markdown("### 6️⃣ 6 Features")

        st.caption("Core System")


    with f3:

        st.markdown("### 🧠 Machine")

        st.caption("Learning")


    with f4:

        st.markdown("### ◈ Smart")

        st.caption("Analysis")


    st.divider()


    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.subheader("📊 Input Data")


    cancer_data = load_breast_cancer()

    dataset = cancer_data.data


    sample_option = st.radio(
        "Choose input method",
        [
            "Use Wisconsin dataset sample",
            "Enter values manually"
        ],
        horizontal=True
    )


    features = []


    # =====================================================
    # SAMPLE INPUT
    # =====================================================

    if sample_option == "Use Wisconsin dataset sample":

        sample_number = st.number_input(
            "Dataset sample number",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
            step=1
        )


        features = dataset[
            int(sample_number)
        ].tolist()


        st.info(
            f"Using Wisconsin dataset sample "
            f"#{int(sample_number)}."
        )


        with st.expander(
            "View 30 medical input features"
        ):

            feature_df = pd.DataFrame(
                {
                    "Feature": FEATURE_NAMES,
                    "Value": features
                }
            )


            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )


    # =====================================================
    # MANUAL INPUT
    # =====================================================

    else:

        st.write(
            "Enter the 30 medical feature values."
        )


        columns = st.columns(3)


        for i, feature_name in enumerate(
            FEATURE_NAMES
        ):

            with columns[i % 3]:

                value = st.number_input(
                    feature_name,
                    value=0.0,
                    format="%.6f",
                    key=f"feature_{i}"
                )

                features.append(value)


    st.divider()


    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button(
        "🚀 START PREDICTION",
        use_container_width=True
    ):

        if len(features) != 30:

            st.error(
                "Exactly 30 medical features are required."
            )

        else:

            with st.spinner(
                "⚛️ Quantum-classical model processing..."
            ):

                try:

                    prediction_response = requests.post(
                        f"{API_URL}/predict",

                        json={
                            "features": features
                        },

                        timeout=120
                    )


                    if prediction_response.status_code == 200:

                        st.session_state.prediction_result = (
                            prediction_response.json()
                        )

                        st.success(
                            "Prediction completed successfully."
                        )

                    else:

                        st.error(
                            f"Prediction failed "
                            f"({prediction_response.status_code})"
                        )

                        try:

                            st.json(
                                prediction_response.json()
                            )

                        except Exception:

                            st.write(
                                prediction_response.text
                            )


                except requests.exceptions.RequestException as e:

                    st.error(
                        "Could not connect to the AI service."
                    )

                    st.code(str(e))


    # =====================================================
    # RESULT
    # =====================================================

    if st.session_state.prediction_result:

        result = st.session_state.prediction_result


        st.divider()

        st.header("📌 Prediction Result")


        prediction = result.get(
            "prediction",
            "Unknown"
        )


        confidence = result.get(
            "confidence",
            0
        )


        malignant_probability = result.get(
            "malignant_probability",
            0
        )


        benign_probability = result.get(
            "benign_probability",
            0
        )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)


        with c1:

            st.metric(
                "Prediction",
                prediction
            )


        with c2:

            st.metric(
                "Confidence",
                f"{confidence}%"
            )


        with c3:

            st.metric(
                "Benign",
                f"{benign_probability}%"
            )


        with c4:

            st.metric(
                "Malignant",
                f"{malignant_probability}%"
            )


        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        st.subheader("📈 Model Probability")


        st.progress(
            max(
                0,
                min(
                    benign_probability / 100,
                    1
                )
            )
        )


        st.caption(
            f"Benign probability: "
            f"{benign_probability}%"
        )


        # -------------------------------------------------
        # TOP FEATURES
        # -------------------------------------------------

        st.subheader(
            "📊 Most Influential Features"
        )


        top_features = result.get(
            "top_features",
            []
        )


        if top_features:

            rows = []


            for item in top_features:

                rows.append(
                    {
                        "Feature": item.get(
                            "feature",
                            ""
                        ),

                        "Contribution": item.get(
                            "contribution",
                            0
                        ),

                        "Direction": item.get(
                            "direction",
                            ""
                        )
                    }
                )


            feature_df = pd.DataFrame(
                rows
            )


            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )


        # -------------------------------------------------
        # AI EXPLANATION
        # -------------------------------------------------

        st.subheader(
            "🧠 AI Explanation"
        )


        explanation = result.get(
            "explanation",
            ""
        )


        if explanation:

            st.info(
                explanation
            )

        else:

            st.warning(
                "AI explanation is unavailable."
            )


        # -------------------------------------------------
        # DISCLAIMER
        # -------------------------------------------------

        st.warning(
            result.get(
                "disclaimer",
                "This is a machine-learning prediction "
                "for research and decision-support purposes. "
                "It is not a medical diagnosis."
            )
        )


# =========================================================
# AI CHAT TAB
# =========================================================

with chat_tab:

    st.header("💬 AI Assistant")

    st.write(
        "Talk with the HEXAMIND AI about quantum "
        "computing, machine learning, programming, "
        "science, technology, education, or this project."
    )


    st.divider()


    # -----------------------------------------------------
    # CLEAR CHAT
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.chat_messages = []

        st.rerun()


    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # WELCOME
    # -----------------------------------------------------

    if not st.session_state.chat_messages:

        st.info(
            "🤖 Hello! I'm the HEXAMIND AI assistant. "
            "Ask me anything about quantum computing, "
            "machine learning, programming, science, "
            "or your project."
        )


    # -----------------------------------------------------
    # CHAT INPUT
    # -----------------------------------------------------

    user_message = st.chat_input(
        "Ask me anything..."
    )


    # -----------------------------------------------------
    # SEND MESSAGE
    # -----------------------------------------------------

    if user_message:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )


        with st.chat_message("user"):

            st.markdown(
                user_message
            )


        with st.chat_message("assistant"):

            with st.spinner(
                "🧠 Thinking..."
            ):

                try:

                    chat_response = requests.post(
                        f"{API_URL}/chat",

                        json={
                            "messages":
                            st.session_state.chat_messages
                        },

                        timeout=120
                    )


                    if chat_response.status_code == 200:

                        chat_data = (
                            chat_response.json()
                        )


                        assistant_reply = (
                            chat_data.get(
                                "reply",
                                "Sorry, I couldn't generate a response."
                            )
                        )


                        st.markdown(
                            assistant_reply
                        )


                        st.session_state.chat_messages.append(
                            {
                                "role": "assistant",
                                "content": assistant_reply
                            }
                        )


                    else:

                        st.error(
                            "AI service returned an error."
                        )


                        try:

                            st.json(
                                chat_response.json()
                            )

                        except Exception:

                            st.write(
                                chat_response.text
                            )


                except requests.exceptions.RequestException as e:

                    st.error(
                        "Could not connect to the AI service."
                    )

                    st.caption(
                        str(e)
                    )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.markdown(
    '<div class="hex-footer">'
    'HEXAMIND TEAM • 6 Core Features • 1 Intelligent System'
    '</div>',
    unsafe_allow_html=True
)