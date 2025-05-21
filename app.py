import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Streamlit page config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# Default CSV path (your uploaded file)
default_csv_path = "Mall_Customers.csv"

# Upload CSV or use default
uploaded_file = st.file_uploader("Upload your own Mall_Customers.csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded!")
else:
    if os.path.exists(default_csv_path):
        df = pd.read_csv(default_csv_path)
        st.info("ℹ️ Using default sample dataset (Mall_Customers.csv)")
    else:
        st.warning("⚠️ Please upload a valid Mall_Customers.csv file.")
        st.stop()

# Drop CustomerID if exists
if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

st.subheader("Preview of Dataset")
st.dataframe(df.head())

# Feature selection
numeric_features = df.select_dtypes(include=np.number).columns.tolist()
selected_features = st.multiselect("Select features for clustering:", options=numeric_features,
                                   default=["Annual Income (k$)", "Spending Score (1-100)"])

if len(selected_features) < 2:
    st.warning("Please select at least two features for clustering.")
    st.stop()

# Standardize features
X = df[selected_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose number of clusters
k = st.slider("Select number of clusters (k):", min_value=2, max_value=10, value=5)

# KMeans clustering
kmeans = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Plot clusters
st.subheader("Cluster Visualization")

fig, ax = plt.subplots()
sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]],
                hue=df["Cluster"], palette="Set2", s=100, ax=ax)
plt.title("Customer Segments")
st.pyplot(fig)

# Show cluster centroids (optional)
st.subheader("Cluster Averages")
st.dataframe(df.groupby("Cluster")[selected_features].mean().round(1))

st.success("🎉 Clustering complete!")
