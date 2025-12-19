import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Title
# -----------------------------
st.title("Sustainable Waste Management Analysis")

# -----------------------------
# Upload file
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload sustainable_waste_management_dataset_2024.csv",
    type=["csv"]
)

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    numeric_cols = [
        'population', 'waste_kg', 'recyclable_kg', 'organic_kg',
        'collection_capacity_kg', 'temp_c', 'rain_mm',
        'is_weekend', 'is_holiday', 'recycling_campaign'
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna()
    return df

if uploaded_file is None:
    st.stop()

df = load_data(uploaded_file)

# -----------------------------
# Data Preview
# -----------------------------
st.header("Dataset Preview")
st.dataframe(df.head(10))
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# -----------------------------
# Data Visualization
# -----------------------------
st.header("Data Visualization")

fig1, ax1 = plt.subplots()
ax1.hist(df['waste_kg'], bins=20)
ax1.set_xlabel("Waste (kg)")
ax1.set_ylabel("Frequency")
ax1.set_title("Waste Distribution")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
ax2.hist(df['recyclable_kg'], bins=20)
ax2.set_xlabel("Recyclable Waste (kg)")
ax2.set_ylabel("Frequency")
ax2.set_title("Recyclable Waste Distribution")
st.pyplot(fig2)

# -----------------------------
# Linear Regression Model
# -----------------------------
st.header("Linear Regression Model")
degree = st.slider("Polynomial Degree", 1)
test_size = st.slider("Test Size", 0.1)

X = df[
    [
        'population', 'recyclable_kg', 'organic_kg',
        'temp_c', 'rain_mm',
        'is_weekend', 'is_holiday', 'recycling_campaign'
    ]
]

y = df['waste_kg']

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
# Evaluation
# -----------------------------
st.header("Model Evaluation")
st.write("MSE:", mean_squared_error(y_test, y_pred))
st.write("R-square:", r2_score(y_test, y_pred))

# -----------------------------
# Prediction vs Actual
# -----------------------------
fig3, ax3 = plt.subplots()
ax3.scatter(y_test, y_pred)
ax3.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    '--'
)
ax3.set_xlabel("Actual Waste (kg)")
ax3.set_ylabel("Predicted Waste (kg)")
ax3.set_title("Predicted vs Actual Waste")
st.pyplot(fig3)
