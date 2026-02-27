st.write("Version 2 deployed 🚀")
import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Retail Analytics", page_icon="🛒", layout="wide")
st.title("🛒 AI Retail Analytics")
st.caption("Upload your data to run Customer Segmentation (Clustering) + Market Basket Analysis (Association Rules).")

# ---------- Helpers ----------
def read_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Upload CSV or Excel.")

def plot_pca_scatter(X_scaled: np.ndarray, labels: np.ndarray, title: str):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    dfp = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels})

    fig, ax = plt.subplots()
    for c in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=18, label=str(c))
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)

def safe_cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray):
    # Metrics only valid when >=2 clusters and no single-cluster case.
    unique = np.unique(labels)
    if len(unique) < 2:
        return None, None

    # For DBSCAN, ignore noise label -1 for metrics (common practice)
    if -1 in unique:
        mask = labels != -1
        if mask.sum() < 3:
            return None, None
        lab = labels[mask]
        Xs = X_scaled[mask]
        if len(np.unique(lab)) < 2:
            return None, None
        return silhouette_score(Xs, lab), davies_bouldin_score(Xs, lab)

    return silhouette_score(X_scaled, labels), davies_bouldin_score(X_scaled, labels)

def df_to_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# ---------- Tabs ----------
tab1, tab2 = st.tabs(["👥 Customer Segmentation (Clustering)", "🧺 Association Rules (Market Basket)"])

# =========================
# TAB 1: CLUSTERING
# =========================
with tab1:
    st.subheader("👥 Customer Segmentation (Clustering)")
    st.write("Upload a **customer-level** dataset (one row per customer) with numeric features (e.g., RFM).")

    up = st.file_uploader("Upload customer-level CSV/XLSX", type=["csv", "xlsx", "xls"], key="cust")
    if up is None:
        st.info("Upload a file to begin.")
    else:
        df = read_file(up)
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

        st.markdown("### 1) Select numeric features")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("Need at least 2 numeric columns for clustering. Add numeric features (e.g., Recency, Frequency, Monetary).")
            st.stop()

        features = st.multiselect(
            "Choose numeric columns",
            numeric_cols,
            default=numeric_cols[: min(6, len(numeric_cols))]
        )
        if len(features) < 2:
            st.warning("Select at least 2 numeric columns.")
            st.stop()

        X = df[features].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        st.markdown("### 2) Choose algorithm")
        algo = st.selectbox("Clustering algorithm", ["K-Means", "Hierarchical", "GMM", "DBSCAN"])

        labels = None
        model_name = algo

        if algo == "K-Means":
            k = st.slider("k (clusters)", 2, 12, 4)
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X_scaled)

        elif algo == "Hierarchical":
            k = st.slider("clusters", 2, 12, 4)
            linkage = st.selectbox("linkage", ["ward", "complete", "average", "single"])
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X_scaled)

        elif algo == "GMM":
            k = st.slider("components", 2, 12, 4)
            cov = st.selectbox("covariance_type", ["full", "tied", "diag", "spherical"])
            model = GaussianMixture(n_components=k, covariance_type=cov, random_state=42)
            labels = model.fit_predict(X_scaled)

        elif algo == "DBSCAN":
            eps = st.slider("eps", 0.1, 5.0, 0.5)
            min_samples = st.slider("min_samples", 2, 30, 5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

        st.markdown("### 3) Evaluation (Dissertation-ready)")
        sil, dbi = safe_cluster_metrics(X_scaled, labels)
        c1, c2, c3 = st.columns(3)
        c1.metric("Algorithm", model_name)
        c2.metric("Silhouette", "N/A" if sil is None else f"{sil:.3f}")
        c3.metric("Davies–Bouldin", "N/A" if dbi is None else f"{dbi:.3f}")

        st.markdown("### 4) Cluster summary")
        out = X.copy()
        out["cluster"] = labels
        st.write("Cluster counts:")
        st.dataframe(out["cluster"].value_counts().reset_index().rename(columns={"index": "cluster", "cluster": "count"}),
                     use_container_width=True)

        st.write("Mean feature values per cluster:")
        st.dataframe(out.groupby("cluster")[features].mean().round(3), use_container_width=True)

        st.markdown("### 5) PCA visualization")
        plot_pca_scatter(X_scaled, labels, f"{model_name} — PCA Scatter")

        st.download_button(
            "⬇️ Download clustered data (CSV)",
            data=df_to_download(out),
            file_name="customer_clusters.csv",
            mime="text/csv"
        )

# =========================
# TAB 2: ASSOCIATION RULES
# =========================
with tab2:
    st.subheader("🧺 Association Rules (Market Basket Analysis)")
    st.write("Upload a **transaction-level** dataset: one row per purchased item. You must select a TransactionID column and an Item column.")

    up2 = st.file_uploader("Upload transactions CSV/XLSX", type=["csv", "xlsx", "xls"], key="tx")
    if up2 is None:
        st.info("Upload a file to begin.")
    else:
        df = read_file(up2)
        st.write("Preview:")
        st.dataframe(df.head(), use_container_width=True)

        cols = df.columns.tolist()
        if len(cols) < 2:
            st.error("Dataset needs at least 2 columns.")
            st.stop()

        st.markdown("### 1) Choose columns")
        tx_col = st.selectbox("Transaction ID column (e.g., InvoiceNo)", cols)
        item_col = st.selectbox("Item column (e.g., Description / Product)", cols)

        df2 = df[[tx_col, item_col]].dropna().copy()
        df2[item_col] = df2[item_col].astype(str).str.strip()

        baskets = df2.groupby(tx_col)[item_col].apply(list).tolist()

        st.write(f"Transactions: {len(baskets):,}")
        if len(baskets) == 0:
            st.error("No transactions found after cleaning. Check your selected columns.")
            st.stop()

        st.markdown("### 2) Build one-hot basket matrix")
        te = TransactionEncoder()
        arr = te.fit(baskets).transform(baskets)
        onehot = pd.DataFrame(arr, columns=te.columns_)

        st.markdown("### 3) Choose algorithm + thresholds")
        algo = st.selectbox("Frequent itemset algorithm", ["Apriori", "FP-Growth"])
        min_support = st.slider("min_support", 0.001, 0.2, 0.02)
        min_conf = st.slider("min_confidence", 0.01, 1.0, 0.2)
        min_lift = st.slider("min_lift", 0.5, 5.0, 1.0)

        if algo == "Apriori":
            freq = apriori(onehot, min_support=min_support, use_colnames=True)
        else:
            freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)

        if len(freq) == 0:
            st.warning("No frequent itemsets found. Try lowering min_support.")
            st.stop()

        freq = freq.sort_values("support", ascending=False)
        st.write("Top frequent itemsets:")
        st.dataframe(freq.head(20), use_container_width=True)

        st.markdown("### 4) Generate rules")
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        rules = rules[rules["lift"] >= min_lift].sort_values(["lift", "confidence"], ascending=False)

        def fs_to_str(x): return ", ".join(sorted(list(x)))
        rules_disp = rules.copy()
        rules_disp["antecedents"] = rules_disp["antecedents"].apply(fs_to_str)
        rules_disp["consequents"] = rules_disp["consequents"].apply(fs_to_str)
        rules_disp = rules_disp[["antecedents", "consequents", "support", "confidence", "lift"]]

        st.write(f"Rules found: {len(rules_disp):,}")
        st.dataframe(rules_disp.head(50), use_container_width=True)

        st.download_button(
            "⬇️ Download rules (CSV)",
            data=df_to_download(rules_disp),
            file_name="association_rules.csv",
            mime="text/csv"
        )
