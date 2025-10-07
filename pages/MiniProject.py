#Import Library
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib as jl

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score




# -----------------------------
# Safe Path Helper
# -----------------------------
def get_path(*relative_parts):
    """Build absolute path relative to this script, safe for Streamlit Cloud."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(base_dir, "..", *relative_parts))


# -----------------------------
# Load Datasets (safe path)
# -----------------------------
try:
    df_clean_path = get_path("resource", "miniproject", "DatasetwithActual.xlsx")
    df_clean = pd.read_excel(df_clean_path)
except FileNotFoundError:
    st.error(f"❌ Dataset not found: {df_clean_path}")
    st.stop()

try:
    train_df_path = get_path("resource", "miniproject", "Dataset.xlsx")
    train_df = pd.read_excel(train_df_path)
except FileNotFoundError:
    st.error(f"❌ Training dataset not found: {train_df_path}")
    st.stop()


# -----------------------------
# Prepare Model
# -----------------------------
X = df_clean[["videoTitle", "videoDescription", "videoCategoryLabel", "durationSec"]]
y = df_clean["hot"]

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("title_tfidf", TfidfVectorizer(stop_words="english", max_features=1000), "videoTitle"),
        ("desc_tfidf", TfidfVectorizer(stop_words="english", max_features=1000), "videoDescription"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["videoCategoryLabel"]),
        ("duration", StandardScaler(), ["durationSec"]),
    ]
)

# Full pipeline: preprocessing + classifier
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train
model.fit(X_train, y_train)


# -----------------------------
# UI: Title and Style
# -----------------------------
st.markdown("""
    <style>
    .card {
        background-color: #cc0000;
        padding: 5px;
        border-radius: 30px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        margin-bottom: 20px;
        text-align: center;
        color : white;
    }
    label, .stTextInput label, .stNumberInput label, .stTextArea label {
        text-align: left; 
        display: block; 
    }
    .stExpander{
        background-color : #cc0000 ;
        color : white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/b/b8/YouTube_Logo_2017.svg",
    use_container_width=True
)
st.markdown('<div class="card"><h3>📊 Analytics Tendency</h3></div>', unsafe_allow_html=True)
st.title("📝 Enter video details to predict popularity")


# -----------------------------
# Upload or Manual Input
# -----------------------------
isUploadFile = st.toggle("Upload File ?" )

if isUploadFile:
    with st.expander("Example of Dataset"):
        example_path = get_path("resource", "miniproject", "Example.xlsx")
        if os.path.exists(example_path):
            example_df = pd.read_excel(example_path)
            st.dataframe(example_df)
        else:
            st.warning("Example.xlsx not found in resource/miniproject/")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
else:
    uploaded_file = None
    videoTitle = st.text_input("🎬 Video Title 🎬")
    videoDescription = st.text_area("🖊️ Description 🖊️")
    videoCategoryLabel = st.selectbox("📂 Category 📂", [
        'People & Blogs', 'Entertainment', 'Science & Technology',
        'Howto & Style', 'Education', 'Pets & Animals', 'Gaming', 'Sports',
        'News & Politics', 'Music', 'Film & Animation'
    ])
    durationSec = st.number_input("⏱️ Duration (seconds) ⏱️", min_value=1, step=1)


# -----------------------------
# Prediction Section
# -----------------------------
button = st.button("Predict")

if button:
    if uploaded_file is not None:
        st.write("✅ File uploaded:", uploaded_file.name)
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        df['Predict'] = model.predict(df)
        st.dataframe(df)
    else:
        data = pd.DataFrame([{
            "videoTitle": videoTitle,
            "videoDescription": videoDescription,
            "videoCategoryLabel": videoCategoryLabel,
            "durationSec": durationSec
        }])
        predict_answer = model.predict(data)
        if predict_answer == 1:
            st.success("Your video is trend 🔥")
        else:
            st.warning("Your video is not trend 😥")


# -----------------------------
# Show Dataset Section
# -----------------------------
show_data = st.toggle("Show training dataset")

if show_data:
    set_index = st.checkbox("Set Index ?", value=False)
    if set_index:
        set_index_columns = st.selectbox("How would you like to set index?", train_df.columns)
        columns = st.multiselect(
            "Select columns to display:",
            options=train_df.columns.tolist(),
            default=train_df.columns.tolist()[:4]
        )
        if columns:
            st.dataframe(train_df[columns], use_container_width=True)
        else:
            st.warning("⚠ Please select at least one column!")
    else:
        st.dataframe(train_df, use_container_width=True)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            f"""<div class="card">
            <h5>📑 Dataset</h5><p style="font-size:18px;">Rows : {df_clean.shape[0]}<br>Columns : {df_clean.shape[1]}</p></div>""",
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""<div class="card">
            <h5>🎯 Accuracy</h5><p style="font-size:36px">{acc *100:.2f} % </p></div>""",
            unsafe_allow_html=True
        )
