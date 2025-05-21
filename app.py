import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except ImportError:
    st.error("🚨 Missing scikit-learn library. Please install it using: `pip install scikit-learn`")
    st.stop()

# Streamlit page configuration
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# File upload or use default dataset
default_csv_path = "Mall_Customers.csv"  # Assuming the file is in the same directory as app.py
uploaded_file = st.file_uploader("Upload your own Mall_Customers.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded!")
else:
    if os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
        st.info("ℹ️ Using default sample dataset (Mall_Customers.csv)")
    else:
        st.warning("⚠️ Please upload a valid Mall_Customers.csv file or ensure it exists in the project directory.")
        st.stop()

# Drop CustomerID if it exists
if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

# Display dataset preview
st.subheader("Preview of Dataset")
st.dataframe(df.head())

# Feature selection for clustering
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
selected_features = st.multiselect(
    "Select features for clustering:",
    options=numeric_features,
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

# Validate feature selection
if len(selected_features) < 2:
    st.warning("⚠️ Please select at least two features for clustering.")
    st.stop()

# Prepare data for clustering
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Select number of clusters
k = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=5)

# Perform KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Save the KMeans model
model_path = "kmeans_model.pkl"
joblib.dump(kmeans, model_path)
st.info(f"💾 KMeans model saved to {model_path}")

# Visualize clusters
st.subheader("Cluster Visualization")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x=df[selected_features[0]],
    y=df[selected_features[1]],
    hue=df["Cluster"],
    palette="Set2",
    s=100,
    ax=ax
)
plt.title("Customer Segments")
st.pyplot(fig)

# Display cluster averages
st.subheader("Cluster Averages")
st.dataframe(df.groupby("Cluster")[selected_features].mean().round(1))

st.success("🎉 Clustering complete!")
