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
st.caption("Premium dashboard: RFM Segmentation + Market Basket Analysis (Online Retail Dataset Ready)")


# =========================================================
# HELPERS
# =========================================================
def read_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Upload CSV or Excel only.")


def load_demo_or_upload(key: str, label: str) -> pd.DataFrame:
    """
    Loads a demo dataset from repo (data/Online Retail.xlsx) OR allows file upload.
    """
    demo_path = Path(__file__).resolve().parent / "data" / "Online Retail.xlsx"

    with st.sidebar:
        st.markdown("### 📂 Dataset")
        use_demo = st.checkbox("✅ Use demo dataset (Online Retail)", value=True, key=f"demo_{key}")
        up = None if use_demo else st.file_uploader(label, type=["csv", "xlsx", "xls"], key=key)

    if use_demo:
        if not demo_path.exists():
            st.error("Demo dataset not found. Upload it to GitHub at: `data/Online Retail.xlsx`")
            st.stop()
        return pd.read_excel(demo_path)

    if up is None:
        st.info("Upload the Online Retail dataset to begin, or turn ON the demo dataset checkbox.")
        st.stop()

    return read_file(up)


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

    # For DBSCAN, ignore noise label -1 for metrics
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


def quality_bar(label: str, value: float, mode="high"):
    st.markdown(f"**{label}**")
    if value is None:
        st.info("Not available for this result.")
        return

    if mode == "high":
        # silhouette in [-1, 1] -> [0, 1]
        normalized = (value + 1) / 2
    else:
        # DBI: lower better; squish to [0,1]
        normalized = 1 / (1 + max(value, 0))

    normalized = float(np.clip(normalized, 0, 1))
    st.progress(normalized)
    st.caption(f"Score: {value:.3f}")


def plot_pca_scatter(X_scaled: np.ndarray, labels: np.ndarray, title: str, label_map: dict | None = None):
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    dfp = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "cluster": labels})

    fig, ax = plt.subplots()
    for c in sorted(dfp["cluster"].unique()):
        sub = dfp[dfp["cluster"] == c]
        name = label_map.get(c) if label_map else None
        legend_label = f"{c} - {name}" if name else str(c)
        ax.scatter(sub["PC1"], sub["PC2"], s=22, label=legend_label)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)


def rules_strength_bar(conf: float, lift: float):
    lift_norm = min(lift, 5.0) / 5.0
    strength = 0.6 * conf + 0.4 * lift_norm
    strength = float(np.clip(strength, 0, 1))
    st.progress(strength)
    st.caption(f"Strength ≈ {strength:.2f} (confidence={conf:.2f}, lift={lift:.2f})")


def name_clusters(rfm_df: pd.DataFrame, cluster_col="cluster") -> pd.DataFrame:
    """
    Auto-name clusters using cluster means.
    Higher Monetary/Frequency = better.
    Lower Recency = better (more recent).
    """
    out = rfm_df.copy()
    summary = out.groupby(cluster_col)[["Recency", "Frequency", "Monetary"]].mean().reset_index()

    rec_q75 = summary["Recency"].quantile(0.75)
    mon_q75 = summary["Monetary"].quantile(0.75)
    freq_q75 = summary["Frequency"].quantile(0.75)

    names = {}
    for _, row in summary.iterrows():
        c = int(row[cluster_col])
        rec, freq, mon = row["Recency"], row["Frequency"], row["Monetary"]

        if (mon >= mon_q75) and (freq >= freq_q75) and (rec <= rec_q75):
            label = "VIP / Champions"
        elif (freq >= freq_q75) and (rec <= rec_q75):
            label = "Loyal Customers"
        elif (rec > rec_q75) and (mon >= mon_q75):
            label = "High-Value At Risk"
        elif (rec > rec_q75) and (freq >= freq_q75):
            label = "Loyal But Inactive"
        elif (rec <= rec_q75) and (freq <= summary["Frequency"].median()):
            label = "New / Potential"
        else:
            label = "Budget / Occasional"

        names[c] = label

    out["cluster_name"] = out[cluster_col].apply(lambda x: names.get(int(x), "Unknown"))
    return out


def cluster_recommendation(name: str) -> str:
    n = name.lower()
    if "vip" in n or "champion" in n:
        return "🎁 Offer loyalty rewards, early access to products, and premium membership programs."
    if "loyal" in n and "inactive" not in n:
        return "💎 Maintain engagement with personalized offers and cross-selling campaigns."
    if "at risk" in n:
        return "📩 Launch win-back campaigns with discounts and personalized reminders."
    if "inactive" in n:
        return "🔔 Send reactivation emails and limited-time offers to bring them back."
    if "new" in n or "potential" in n:
        return "🌱 Provide onboarding offers and highlight best-selling products."
    if "budget" in n or "occasional" in n:
        return "🪙 Promote bundle deals and price-sensitive campaigns."
    return "📊 Monitor behavior and personalize marketing based on engagement patterns."


def cluster_badge(name: str) -> str:
    n = name.lower()
    if "vip" in n or "champion" in n:
        return "✅"
    if "at risk" in n:
        return "⚠️"
    if "inactive" in n:
        return "⏳"
    if "loyal" in n:
        return "💎"
    if "new" in n or "potential" in n:
        return "🌱"
    if "budget" in n or "occasional" in n:
        return "🪙"
    return "🏷️"


def cluster_profit_table(rfm_out: pd.DataFrame, margin: float, cost_per_customer: float, target_rate: float):
    total_rev = float(rfm_out["Monetary"].sum())

    t = (
        rfm_out.groupby(["cluster", "cluster_name"], dropna=False)
        .agg(
            customers=("CustomerID", "count"),
            revenue=("Monetary", "sum"),
            avg_revenue=("Monetary", "mean"),
            avg_recency=("Recency", "mean"),
            avg_freq=("Frequency", "mean"),
        )
        .reset_index()
    )

    t["revenue_pct"] = np.where(total_rev > 0, (t["revenue"] / total_rev) * 100, 0.0)
    t["gross_profit"] = t["revenue"] * margin

    t["targeted_customers"] = (t["customers"] * target_rate).round().astype(int)
    t["campaign_cost"] = t["targeted_customers"] * cost_per_customer

    t["net_profit_baseline"] = t["gross_profit"] - t["campaign_cost"]
    t["rank_by_revenue"] = t["revenue"].rank(ascending=False, method="dense").astype(int)

    return t.sort_values("revenue", ascending=False)


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
# CUSTOMER SEGMENTATION: RFM + CLUSTERING + BUSINESS LAYER
# =========================================================
if page.startswith("👥"):
    st.markdown("## 👥 Customer Segmentation (RFM + Clustering)")
    st.write("Uses the Online Retail dataset to build **RFM** features and cluster customers into actionable segments.")

    df = load_demo_or_upload("rfm_upload", "Upload Online Retail CSV/XLSX")

    required = ["InvoiceNo", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Please upload the correct Online Retail dataset.")
        st.stop()

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

    # Build RFM
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

    st.markdown("### 🧾 RFM Table")
    st.dataframe(rfm.head(10), use_container_width=True)

    # Sidebar clustering controls
    with st.sidebar:
        st.markdown("### 🧠 Clustering")
        algo = st.selectbox("Algorithm", ["K-Means", "Hierarchical", "GMM", "DBSCAN"])

        params = {}
        if algo in ["K-Means", "Hierarchical", "GMM"]:
            params["k"] = st.slider("Clusters/Components", 2, 12, 4)

        if algo == "Hierarchical":
            params["linkage"] = st.selectbox("Linkage", ["ward", "complete", "average", "single"])

        if algo == "GMM":
            params["cov"] = st.selectbox("Covariance Type", ["full", "tied", "diag", "spherical"])

        if algo == "DBSCAN":
            params["eps"] = st.slider("eps", 0.1, 5.0, 0.6)
            params["min_samples"] = st.slider("min_samples", 2, 40, 5)

        run = st.button("🚀 Run Clustering", use_container_width=True)

    if not run:
        st.info("Choose settings in sidebar and click **Run Clustering**.")
        st.stop()

    # Fit clustering
    features = ["Recency", "Frequency", "Monetary"]
    X = rfm[features].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if algo == "K-Means":
        model = KMeans(n_clusters=params["k"], random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)
    elif algo == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=params["k"], linkage=params["linkage"])
        labels = model.fit_predict(X_scaled)
    elif algo == "GMM":
        model = GaussianMixture(n_components=params["k"], covariance_type=params["cov"], random_state=42)
        labels = model.fit_predict(X_scaled)
    else:
        model = DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
        labels = model.fit_predict(X_scaled)

    rfm_out = rfm.copy()
    rfm_out["cluster"] = labels
    rfm_out = name_clusters(rfm_out, cluster_col="cluster")

    sil, dbi = safe_cluster_metrics(X_scaled, labels)

    st.markdown("### ✅ Clustering Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Algorithm", algo)
    m2.metric("Silhouette", "N/A" if sil is None else f"{sil:.3f}")
    m3.metric("Davies–Bouldin", "N/A" if dbi is None else f"{dbi:.3f}")

    st.markdown("### 📊 Quality Bars")
    c1, c2 = st.columns(2)
    with c1: quality_bar("Silhouette (higher is better)", sil, mode="high")
    with c2: quality_bar("Davies–Bouldin (lower is better)", dbi, mode="low")

    st.markdown("### 🏷️ Cluster Cards & Recommended Actions")
    cards_df = (
        rfm_out.groupby(["cluster", "cluster_name"])
        .agg(
            customers=("CustomerID", "count"),
            avg_recency=("Recency", "mean"),
            avg_freq=("Frequency", "mean"),
            avg_mon=("Monetary", "mean"),
        )
        .reset_index()
        .sort_values("avg_mon", ascending=False)
    )

    cols = st.columns(min(3, max(1, len(cards_df))))
    for i, row in enumerate(cards_df.itertuples(index=False)):
        badge = cluster_badge(row.cluster_name)
        subtitle = (
            f"Customers: {row.customers:,}<br>"
            f"Recency: {row.avg_recency:.0f} days<br>"
            f"Frequency: {row.avg_freq:.1f}<br>"
            f"Monetary: £{row.avg_mon:,.0f}"
        )
        recommendation = cluster_recommendation(row.cluster_name)

        with cols[i % len(cols)]:
            st.markdown(
                f"""
                <div style="
                    padding:16px; border-radius:16px;
                    border:1px solid rgba(0,0,0,0.08);
                    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(240,248,255,0.9));
                    box-shadow: 0 8px 20px rgba(0,0,0,0.08);
                    min-height:220px;
                ">
                    <div style="font-size:18px; font-weight:700;">
                        {badge} {row.cluster_name} (Cluster {row.cluster})
                    </div>
                    <div style="font-size:13px; margin-top:10px;">
                        {subtitle}
                    </div>
                    <div style="margin-top:12px; font-size:13px; font-weight:600;">
                        Recommended Action:
                    </div>
                    <div style="font-size:13px; margin-top:4px;">
                        {recommendation}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("### 🗺️ PCA Visualization")
    label_map = rfm_out.groupby("cluster")["cluster_name"].first().to_dict()
    plot_pca_scatter(X_scaled, labels, f"{algo} — RFM PCA Scatter", label_map=label_map)

    st.markdown("### 📥 Downloads")
    download_csv_button(rfm_out, "⬇️ Download RFM + Clustered Customers (CSV)", "rfm_clusters.csv")


# =========================================================
# ASSOCIATION RULES: MARKET BASKET
# =========================================================
else:
    st.markdown("## 🧺 Association Rules (Market Basket)")
    st.write("Uses the Online Retail dataset to generate association rules using Apriori / FP-Growth.")

    df = load_demo_or_upload("mba_upload", "Upload Online Retail CSV/XLSX")

    required = ["InvoiceNo", "Description"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Please upload the correct Online Retail dataset.")
        st.stop()

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

    freq = freq.sort_values("support", ascending=False)

    st.markdown("### 📦 Top Frequent Itemsets")
    st.dataframe(freq.head(20), use_container_width=True)

    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["lift"] >= min_lift].sort_values(["lift", "confidence"], ascending=False)

    if len(rules) == 0:
        st.warning("No rules match thresholds. Try lowering min_lift or min_confidence.")
        st.stop()

    def fs_to_str(x): return ", ".join(sorted(list(x)))

    rules_disp = rules.copy()
    rules_disp["antecedents"] = rules_disp["antecedents"].apply(fs_to_str)
    rules_disp["consequents"] = rules_disp["consequents"].apply(fs_to_str)
    rules_disp = rules_disp[["antecedents", "consequents", "support", "confidence", "lift"]]

    st.markdown("### 🔥 Top Rules")
    st.write(f"Rules found: **{len(rules_disp):,}**")
    st.dataframe(rules_disp.head(50), use_container_width=True)

    st.markdown("### 📊 Rule Strength (Progress Bar)")
    top_rule = rules_disp.iloc[0]
    st.write(f"**Top Rule:** `{top_rule['antecedents']}` → `{top_rule['consequents']}`")
    rules_strength_bar(float(top_rule["confidence"]), float(top_rule["lift"]))

    st.markdown("### 📥 Download")
    download_csv_button(rules_disp, "⬇️ Download Association Rules (CSV)", "association_rules.csv")
