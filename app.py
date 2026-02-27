import hashlib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="AI Retail Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# AUTH (Simple, stable)
# =========================
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Change these if you want
APP_USERNAME = "AI"
APP_PASSWORD_HASH = _hash("1234")  # password = 1234


def login_required() -> bool:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    with st.sidebar:
        st.markdown("## 🔒 Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        btn = st.button("Login", use_container_width=True)

        if btn:
            if u == APP_USERNAME and _hash(p) == APP_PASSWORD_HASH:
                st.session_state.logged_in = True
                st.success("✅ Logged in")
            else:
                st.error("❌ Wrong username or password")

    return st.session_state.logged_in


# =========================
# HELPERS
# =========================
def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported format. Upload CSV or Excel.")


def download_csv_button(df: pd.DataFrame, label: str, filename: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=True
    )


def safe_cluster_metrics(X_scaled: np.ndarray, labels: np.ndarray):
    """
    Returns (silhouette, dbi) or (None, None) when not valid.
    For DBSCAN, ignores noise (-1) for metrics.
    """
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
        ax.scatter(sub["PC1"], sub["PC2"], s=20, label=f"{c}")
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)


def progress_bar_from_value(label: str, value: float, best="high"):
    """
    best="high" means higher is better, best="low" means lower is better.
    Value is clamped to [0,1] for progress display.
    """
    st.markdown(f"**{label}**")
    if value is None:
        st.info("Not available for this clustering result.")
        return

    if best == "high":
        # silhouette usually in [-1, 1]
        normalized = (value + 1) / 2  # map [-1,1] -> [0,1]
    else:
        # dbi: lower is better, no fixed upper bound, so use a simple transform
        normalized = 1 / (1 + max(value, 0))

    normalized = float(np.clip(normalized, 0, 1))
    st.progress(normalized)
    st.caption(f"Score: {value:.3f}")


def rules_strength_bar(conf: float, lift: float):
    """
    Simple visual indicator for rule strength using both lift and confidence.
    """
    # Normalize: confidence in [0,1]; lift often >1, cap at 5
    lift_norm = min(lift, 5.0) / 5.0
    strength = 0.6 * conf + 0.4 * lift_norm
    strength = float(np.clip(strength, 0, 1))
    st.progress(strength)
    st.caption(f"Strength ≈ {strength:.2f} (confidence={conf:.2f}, lift={lift:.2f})")


# =========================
# LOGIN GATE
# =========================
if not login_required():
    st.title("🛒 AI Retail Analytics")
    st.write("Please login from the sidebar to use the app.")
    st.stop()

with st.sidebar:
    st.divider()
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


# =========================
# HEADER
# =========================
st.title("🛒 AI Retail Analytics")
st.caption("Premium dashboard for Customer Segmentation + Market Basket Analysis (Upload ANY dataset).")

# =========================
# SIDEBAR NAV
# =========================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Choose module",
        ["👥 Customer Segmentation (Clustering)", "🧺 Association Rules (Market Basket)"],
        label_visibility="collapsed"
    )
    st.divider()


# ==========================================================
# PAGE 1: CLUSTERING
# ==========================================================
if page.startswith("👥"):
    st.markdown("## 👥 Customer Segmentation (Clustering)")
    st.write("Upload a **customer-level** dataset (1 row per customer). Select **numeric columns** to cluster.")

    with st.sidebar:
        st.markdown("### 📂 Upload (Customer-level)")
        file_cust = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], key="cust_upload")

    if file_cust is None:
        st.info("Upload a customer-level dataset to begin.")
        st.stop()

    df = read_file(file_cust)

    cA, cB = st.columns([1.2, 1])
    with cA:
        st.markdown("### 👀 Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with cB:
        st.markdown("### 📌 Dataset Summary")
        st.write(f"Rows: **{len(df):,}**")
        st.write(f"Columns: **{df.shape[1]}**")
        st.write("Numeric columns detected:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.code(", ".join(numeric_cols) if numeric_cols else "None")

    if len(numeric_cols) < 2:
        st.error("You need at least **2 numeric columns** for clustering. Add numeric features (e.g., RFM).")
        st.stop()

    with st.sidebar:
        st.markdown("### 🧾 Feature Selection")
        features = st.multiselect(
            "Choose numeric columns",
            numeric_cols,
            default=numeric_cols[: min(6, len(numeric_cols))]
        )

        st.markdown("### 🧠 Algorithm")
        algo = st.selectbox("Clustering method", ["K-Means", "Hierarchical", "GMM", "DBSCAN"])

        params = {}
        if algo in ["K-Means", "Hierarchical", "GMM"]:
            params["k"] = st.slider("Clusters / Components", 2, 12, 4)

        if algo == "Hierarchical":
            params["linkage"] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])

        if algo == "GMM":
            params["cov"] = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])

        if algo == "DBSCAN":
            params["eps"] = st.slider("eps", 0.1, 5.0, 0.5)
            params["min_samples"] = st.slider("min_samples", 2, 30, 5)

        st.divider()
        run_btn = st.button("🚀 Run Clustering", use_container_width=True)

    if not run_btn:
        st.info("Choose features and settings in the sidebar, then click **Run Clustering**.")
        st.stop()

    if len(features) < 2:
        st.warning("Select at least **2 numeric columns**.")
        st.stop()

    # Prepare data
    X = df[features].dropna().copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit model
    if algo == "K-Means":
        model = KMeans(n_clusters=params["k"], random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)

    elif algo == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=params["k"], linkage=params["linkage"])
        labels = model.fit_predict(X_scaled)

    elif algo == "GMM":
        model = GaussianMixture(n_components=params["k"], covariance_type=params["cov"], random_state=42)
        labels = model.fit_predict(X_scaled)

    else:  # DBSCAN
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        labels = model.fit_predict(X_scaled)

    # Metrics
    sil, dbi = safe_cluster_metrics(X_scaled, labels)

    # Outputs
    out = X.copy()
    out["cluster"] = labels

    st.markdown("### ✅ Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", algo)
    m2.metric("Silhouette", "N/A" if sil is None else f"{sil:.3f}")
    m3.metric("Davies–Bouldin", "N/A" if dbi is None else f"{dbi:.3f}")

    # Progress bars
    st.markdown("### 📊 Quality Bars")
    q1, q2 = st.columns(2)
    with q1:
        progress_bar_from_value("Silhouette (higher is better)", sil, best="high")
    with q2:
        progress_bar_from_value("Davies–Bouldin (lower is better)", dbi, best="low")

    # Summary
    left, right = st.columns([1, 1])
    with left:
        st.markdown("### 🧾 Cluster Counts")
        st.dataframe(out["cluster"].value_counts().reset_index().rename(columns={"index": "cluster", "cluster": "count"}),
                     use_container_width=True)
    with right:
        st.markdown("### 📌 Cluster Means")
        st.dataframe(out.groupby("cluster")[features].mean().round(3), use_container_width=True)

    # PCA plot
    st.markdown("### 🗺️ PCA Visualization")
    plot_pca_scatter(X_scaled, labels, f"{algo} — PCA Scatter")

    # Download
    st.markdown("### 📥 Download Report")
    download_csv_button(out, "⬇️ Download Clustered Data (CSV)", "customer_clusters.csv")


# ==========================================================
# PAGE 2: ASSOCIATION RULES
# ==========================================================
else:
    st.markdown("## 🧺 Association Rules (Market Basket Analysis)")
    st.write("Upload a **transaction-level** dataset: one row per purchased item (TransactionID + Item).")

    with st.sidebar:
        st.markdown("### 📂 Upload (Transaction-level)")
        file_tx = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx", "xls"], key="tx_upload")

    if file_tx is None:
        st.info("Upload a transaction-level dataset to begin.")
        st.stop()

    df = read_file(file_tx)
    st.markdown("### 👀 Preview")
    st.dataframe(df.head(10), use_container_width=True)

    cols = df.columns.tolist()
    if len(cols) < 2:
        st.error("Dataset must have at least 2 columns.")
        st.stop()

    with st.sidebar:
        st.markdown("### 🧾 Column Mapping")
        tx_col = st.selectbox("Transaction ID column", cols)
        item_col = st.selectbox("Item / Product column", cols)

        st.markdown("### ⚙️ Parameters")
        algo = st.selectbox("Algorithm", ["Apriori", "FP-Growth"])
        min_support = st.slider("min_support", 0.001, 0.2, 0.02)
        min_conf = st.slider("min_confidence", 0.01, 1.0, 0.2)
        min_lift = st.slider("min_lift", 0.5, 5.0, 1.0)

        st.divider()
        run_rules = st.button("🚀 Generate Rules", use_container_width=True)

    if not run_rules:
        st.info("Set parameters in the sidebar and click **Generate Rules**.")
        st.stop()

    # Clean + transactions
    df2 = df[[tx_col, item_col]].dropna().copy()
    df2[item_col] = df2[item_col].astype(str).str.strip()
    baskets = df2.groupby(tx_col)[item_col].apply(list).tolist()

    st.markdown("### ✅ Dataset Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Rows", f"{len(df2):,}")
    s2.metric("Transactions", f"{len(baskets):,}")
    s3.metric("Unique Items", f"{df2[item_col].nunique():,}")

    if len(baskets) == 0:
        st.error("No baskets created. Check selected columns.")
        st.stop()

    # One-hot
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    onehot = pd.DataFrame(arr, columns=te.columns_)

    # Frequent itemsets
    if algo == "Apriori":
        freq = apriori(onehot, min_support=min_support, use_colnames=True)
    else:
        freq = fpgrowth(onehot, min_support=min_support, use_colnames=True)

    if len(freq) == 0:
        st.warning("No frequent itemsets. Lower min_support.")
        st.stop()

    freq = freq.sort_values("support", ascending=False)

    st.markdown("### 📦 Top Frequent Itemsets")
    st.dataframe(freq.head(20), use_container_width=True)

    # Rules
    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift].sort_values(["lift", "confidence"], ascending=False)

    if len(rules) == 0:
        st.warning("No rules found with these thresholds. Try lowering min_lift or min_confidence.")
        st.stop()

    def fs_to_str(x): return ", ".join(sorted(list(x)))

    rules_disp = rules.copy()
    rules_disp["antecedents"] = rules_disp["antecedents"].apply(fs_to_str)
    rules_disp["consequents"] = rules_disp["consequents"].apply(fs_to_str)
    rules_disp = rules_disp[["antecedents", "consequents", "support", "confidence", "lift"]]

    st.markdown("### 🔥 Top Rules")
    st.write(f"Rules found: **{len(rules_disp):,}**")
    st.dataframe(rules_disp.head(50), use_container_width=True)

    # Rule strength UI
    st.markdown("### 📊 Rule Strength (Progress Bar)")
    top_rule = rules_disp.iloc[0]
    st.write(f"**Top Rule:** `{top_rule['antecedents']}` → `{top_rule['consequents']}`")
    rules_strength_bar(float(top_rule["confidence"]), float(top_rule["lift"]))

    # "Feature importance" style insight for rules
    st.markdown("### 🧠 Feature-Importance Style Insight (Top Drivers)")
    st.caption("This highlights which items most strongly lead to other items (based on average lift).")

    tmp = rules_disp.copy()
    # Take first item of antecedents for a simple driver analysis
    tmp["driver"] = tmp["antecedents"].apply(lambda s: s.split(",")[0].strip())
    driver_strength = tmp.groupby("driver")[["lift", "confidence"]].mean().sort_values("lift", ascending=False).head(10)

    st.dataframe(driver_strength.round(3), use_container_width=True)

    # Plot driver lift
    fig, ax = plt.subplots()
    ax.bar(driver_strength.index.astype(str), driver_strength["lift"].values)
    ax.set_title("Top Drivers by Average Lift")
    ax.set_xlabel("Item (Driver)")
    ax.set_ylabel("Average Lift")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### 📥 Download Report")
    download_csv_button(rules_disp, "⬇️ Download Rules (CSV)", "association_rules.csv")
