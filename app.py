import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Retail Analytics", page_icon="🛒", layout="wide")
st.title("🛒 AI Retail Analytics")
st.caption("RFM Segmentation + Market Basket Analysis (with demo dataset)")


# =========================================================
# DEMO DATASET SUPPORT
# =========================================================
DATA_PATH = Path(__file__).parent / "data" / "Online Retail.xlsx"


def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload CSV or Excel only.")


def load_dataset():
    st.sidebar.markdown("## 📂 Dataset")

    use_demo = st.sidebar.checkbox("Use demo dataset (recommended)", value=True)

    uploaded = st.sidebar.file_uploader(
        "Or upload your own CSV/XLSX",
        type=["csv", "xlsx", "xls"]
    )

    if use_demo:
        if not DATA_PATH.exists():
            st.error("❌ Demo dataset not found. Upload it to: `data/Online Retail.xlsx`")
            st.stop()
        df = pd.read_excel(DATA_PATH)
        st.sidebar.success("✅ Loaded demo dataset")
        return df

    if uploaded is None:
        st.info("Enable demo dataset OR upload your own file.")
        st.stop()

    df = read_file(uploaded)
    st.sidebar.success("✅ Uploaded file loaded")
    return df


# =========================================================
# UI HELPERS
# =========================================================
def kpi_card(label: str, value: str):
    st.markdown(
        f"""
        <div style="
            padding:14px; border-radius:14px; border:1px solid rgba(0,0,0,0.08);
            background: rgba(255,255,255,0.7);
            box-shadow: 0 6px 16px rgba(0,0,0,0.06);
        ">
            <div style="font-size:13px; opacity:0.7">{label}</div>
            <div style="font-size:26px; font-weight:700; margin-top:4px;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def download_csv_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )


def safe_cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return None, None

    if -1 in uniq:
        mask = labels != -1
        if mask.sum() < 3:
            return None, None
        lab = labels[mask]
        Xs = X_scaled[mask]
        if len(np.unique(lab)) < 2:
            return None, None
        return silhouette_score(Xs, lab), davies_bouldin_score(Xs, lab)

    return silhouette_score(X_scaled, labels), davies_bouldin_score(X_scaled, labels)


def plot_pca_scatter(X_scaled: np.ndarray, labels: np.ndarray, title: str):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    dfp = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels})

    fig, ax = plt.subplots()
    for c in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == c]
        ax.scatter(sub["PC1"], sub["PC2"], s=20, label=f"Cluster {c}")

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)


# =========================================================
# SIDEBAR NAV
# =========================================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Choose module",
        ["👥 Customer Segmentation (RFM + Clustering)", "🧺 Association Rules (Market Basket)"],
        label_visibility="collapsed"
    )
    st.divider()


# =========================================================
# LOAD DATASET
# =========================================================
df = load_dataset()


required = ["InvoiceNo", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}. Upload the standard Online Retail dataset format.")
    st.stop()


# =========================================================
# CUSTOMER SEGMENTATION (RFM)
# =========================================================
if page.startswith("👥"):
    st.markdown("## 👥 Customer Segmentation (RFM + Clustering)")

    # Clean
    df = df.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"])
    df["CustomerID"] = df["CustomerID"].astype(int)

    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"].astype(float) > 0) & (df["UnitPrice"].astype(float) > 0)]
    df["Revenue"] = df["Quantity"].astype(float) * df["UnitPrice"].astype(float)

    # KPIs
    total_revenue = df["Revenue"].sum()
    customers = df["CustomerID"].nunique()
    invoices = df["InvoiceNo"].nunique()
    items = df["Description"].nunique()

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total Revenue", f"£ {total_revenue:,.0f}")
    with k2: kpi_card("Customers", f"{customers:,}")
    with k3: kpi_card("Invoices", f"{invoices:,}")
    with k4: kpi_card("Unique Items", f"{items:,}")

    st.markdown("### 👀 Preview")
    st.dataframe(df.head(12), use_container_width=True)

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    st.markdown("### 🧾 RFM Table")
    st.dataframe(rfm.head(10), use_container_width=True)

    with st.sidebar:
        st.markdown("### 🧠 Clustering")
        algo = st.selectbox("Algorithm", ["K-Means", "Hierarchical", "GMM", "DBSCAN"])

        params = {}
        if algo in ["K-Means", "Hierarchical", "GMM"]:
            params["k"] = st.slider("Clusters", 2, 12, 4)

        if algo == "DBSCAN":
            params["eps"] = st.slider("eps", 0.1, 5.0, 0.6)
            params["min_samples"] = st.slider("min_samples", 2, 40, 5)

        run = st.button("🚀 Run Clustering", use_container_width=True)

    if not run:
        st.info("Choose settings in sidebar and click **Run Clustering**.")
        st.stop()

    X = rfm[["Recency", "Frequency", "Monetary"]].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algo == "K-Means":
        model = KMeans(n_clusters=params["k"], random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
    elif algo == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=params["k"])
        labels = model.fit_predict(X_scaled)
    elif algo == "GMM":
        model = GaussianMixture(n_components=params["k"], random_state=42)
        labels = model.fit_predict(X_scaled)
    else:
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        labels = model.fit_predict(X_scaled)

    rfm_out = rfm.copy()
    rfm_out["cluster"] = labels

    sil, dbi = safe_cluster_metrics(X_scaled, labels)

    st.markdown("### ✅ Results")
    st.write(f"**Algorithm:** {algo}")
    st.write(f"**Silhouette:** {sil if sil is not None else 'N/A'}")
    st.write(f"**Davies–Bouldin:** {dbi if dbi is not None else 'N/A'}")

    st.dataframe(rfm_out.head(15), use_container_width=True)

    st.markdown("### 🗺️ PCA Scatter Plot")
    plot_pca_scatter(X_scaled, labels, f"{algo} — RFM Clusters (PCA)")

    st.markdown("### 📥 Download Results")
    download_csv_button(rfm_out, "⬇️ Download RFM + Clusters (CSV)", "rfm_clusters.csv")


# =========================================================
# MARKET BASKET ANALYSIS
# =========================================================
else:
    st.markdown("## 🧺 Association Rules (Market Basket)")

    df = df.dropna(subset=["InvoiceNo", "Description"])
    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    st.markdown("### 👀 Preview")
    st.dataframe(df[["InvoiceNo", "Description"]].head(12), use_container_width=True)

    with st.sidebar:
        st.markdown("### ⚙️ Rule Settings")
        algo = st.selectbox("Algorithm", ["Apriori", "FP-Growth"])
        min_support = st.slider("min_support", 0.001, 0.2, 0.02)
        min_conf = st.slider("min_confidence", 0.01, 1.0, 0.2)
        min_lift = st.slider("min_lift", 0.5, 5.0, 1.0)
        run = st.button("🚀 Generate Rules", use_container_width=True)

    if not run:
        st.info("Set parameters in sidebar and click **Generate Rules**.")
        st.stop()

    baskets = df.groupby("InvoiceNo")["Description"].apply(list).tolist()

    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    onehot = pd.DataFrame(arr, columns=te.columns_)

    if algo == "Apriori":
        freq = apriori(onehot, min_support=min_support, use_colnames=True)
    else:
        freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)

    if len(freq) == 0:
        st.warning("No frequent itemsets found. Lower min_support.")
        st.stop()

    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift].sort_values(["lift", "confidence"], ascending=False)

    if len(rules) == 0:
        st.warning("No rules match thresholds. Lower min_lift or min_confidence.")
        st.stop()

    def fs_to_str(x): return ", ".join(sorted(list(x)))

    rules_disp = rules.copy()
    rules_disp["antecedents"] = rules_disp["antecedents"].apply(fs_to_str)
    rules_disp["consequents"] = rules_disp["consequents"].apply(fs_to_str)
    rules_disp = rules_disp[["antecedents", "consequents", "support", "confidence", "lift"]]

    st.markdown("### 🔥 Top Rules")
    st.dataframe(rules_disp.head(50), use_container_width=True)

    st.markdown("### 📥 Download Rules")
    download_csv_button(rules_disp, "⬇️ Download Rules (CSV)", "association_rules.csv")
