import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Neo Arcadia Game Analysis",
    page_icon="🎮",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("🎮 Neo Arcadia Game Session Analysis")
st.markdown("Mini Project #5 — Build a Website with Streamlit")

# -----------------------------
# Sidebar: Upload File
# -----------------------------
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload neo_arcadia_missions.csv",
    type=["csv"]
)

# -----------------------------
# Load Data Function
# -----------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    df['session_minutes'] = pd.to_numeric(df['session_minutes'], errors='coerce')
    df['coins_spent'] = pd.to_numeric(df['coins_spent'], errors='coerce')
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['hour'] = df['start_time'].dt.hour

    df = df.dropna(subset=['session_minutes', 'coins_spent', 'win_flag', 'hour'])
    return df

# -----------------------------
# Stop if no file
# -----------------------------
if uploaded_file is None:
    st.warning("⬅️ กรุณาอัปโหลดไฟล์ neo_arcadia_missions.csv ก่อน")
    st.stop()

df = load_data(uploaded_file)

# -----------------------------
# Sidebar: Model Settings
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

degree = st.sidebar.slider(
    "Polynomial Degree",
    min_value=1,
    max_value=3,
    value=2
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2
)

# -----------------------------
# Section 1: Data Preview
# -----------------------------
st.subheader("📋 Dataset Preview")
st.dataframe(df.head(10))
st.markdown(f"**Dataset Size:** {df.shape[0]} rows × {df.shape[1]} columns")

# -----------------------------
# Section 2: Visualization
# -----------------------------
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['coins_spent'], df['session_minutes'], alpha=0.6)
    ax1.set_xlabel("Coins Spent")
    ax1.set_ylabel("Session Minutes")
    ax1.set_title("Coins Spent vs Session Minutes")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.hist(df['session_minutes'], bins=20)
    ax2.set_xlabel("Session Minutes")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Session Minutes Distribution")
    st.pyplot(fig2)

# -----------------------------
# Section 3: Train Model
# -----------------------------
st.subheader("🤖 Linear Regression Model")

X = df[['coins_spent', 'win_flag', 'hour']]
y = df['session_minutes']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

model = Pipeline([
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('lr', LinearRegression())
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -----------------------------
# Metrics
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

col3, col4 = st.columns(2)
col3.metric("📉 Mean Squared Error (MSE)", f"{mse:.2f}")
col4.metric("📈 R-square (R²)", f"{r2:.2f}")

# -----------------------------
# Section 4: Prediction vs Actual Plot
# -----------------------------
st.subheader("📈 Predicted vs Actual")

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.scatter(y_test, y_pred, alpha=0.7)
ax3.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--', lw=2, label="Perfect Prediction Line"
)
ax3.set_xlabel("Actual Session Minutes")
ax3.set_ylabel("Predicted Session Minutes")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("📌 Mini Project #5 | Built with Streamlit")
