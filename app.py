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


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Retail Analytics", page_icon="🛒", layout="wide")
st.title("🛒 AI Retail Analytics")
st.caption("Premium dashboard: RFM Segmentation + Market Basket Analysis (Online Retail Dataset Ready)")


# =========================================================
# SIMPLE AUTH
# =========================================================
def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

APP_USERNAME = "AI"
APP_PASSWORD_HASH = _hash("1234")  # password: 1234

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

if not login_required():
    st.info("Please login from the sidebar to use the app.")
    st.stop()

with st.sidebar:
    st.divider()
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()


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
    st.write("Upload Online Retail dataset. The app auto-generates **RFM** per customer, then clusters customers and produces executive insights.")

    with st.sidebar:
        st.markdown("### 📂 Upload Dataset")
        up = st.file_uploader("Upload Online Retail CSV/XLSX", type=["csv", "xlsx", "xls"], key="rfm_upload")

    if up is None:
        st.info("Upload the Online Retail dataset to begin.")
        st.stop()

    df = read_file(up)

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

    # Remove cancelled invoices commonly starting with "C"
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Remove invalid quantities/prices
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
    st.markdown("### 🧾 Build RFM Features")
    st.caption("Recency = days since last purchase | Frequency = number of invoices | Monetary = total spend")

    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()

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

    # Add names
    rfm_out = name_clusters(rfm_out, cluster_col="cluster")

    # Metrics
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

    # Cluster cards + actions
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

    # Named summary table
    st.markdown("### 🧾 Cluster Name Summary")
    st.dataframe(
        rfm_out.groupby(["cluster", "cluster_name"])[["Recency", "Frequency", "Monetary"]]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("Monetary", ascending=False),
        use_container_width=True
    )

    # PCA legend with names
    st.markdown("### 🗺️ PCA Visualization")
    label_map = rfm_out.groupby("cluster")["cluster_name"].first().to_dict()
    plot_pca_scatter(X_scaled, labels, f"{algo} — RFM PCA Scatter", label_map=label_map)

    # Profitability + revenue contribution
    st.markdown("## 💰 Cluster Profitability & Revenue Contribution")
    colA, colB, colC = st.columns(3)
    with colA:
        gross_margin = st.slider("Gross margin (%)", 0, 90, 35) / 100.0
    with colB:
        cost_per_customer = st.number_input("Campaign cost per targeted customer (£)", min_value=0.0, value=2.0, step=0.5)
    with colC:
        target_rate = st.slider("Target rate (share of customers targeted)", 0.0, 1.0, 0.30)

    profit_tbl = cluster_profit_table(
        rfm_out=rfm_out,
        margin=gross_margin,
        cost_per_customer=cost_per_customer,
        target_rate=target_rate
    )

    st.markdown("### 📊 Revenue contribution by cluster")
    show_cols = [
        "cluster", "cluster_name", "customers",
        "revenue", "revenue_pct", "gross_profit", "campaign_cost", "net_profit_baseline"
    ]
    st.dataframe(
        profit_tbl[show_cols].round({
            "revenue": 0,
            "revenue_pct": 2,
            "gross_profit": 0,
            "campaign_cost": 0,
            "net_profit_baseline": 0
        }),
        use_container_width=True
    )

    st.markdown("### 🏆 Top clusters by revenue")
    top = profit_tbl.head(3).copy()
    t1, t2, t3 = st.columns(3)
    tcols = [t1, t2, t3]
    for i in range(min(3, len(top))):
        r = top.iloc[i]
        with tcols[i]:
            st.metric(
                label=f"{r['cluster_name']} (C{int(r['cluster'])})",
                value=f"£ {r['revenue']:,.0f}",
                delta=f"{r['revenue_pct']:.1f}% of revenue"
            )

    # ROI simulator
    st.markdown("## 📈 Marketing ROI Simulator")
    st.caption(
        "Estimates ROI using simple assumptions: target a fraction of customers, a % respond, and responders spend more."
    )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        response_rate = st.slider("Response rate (of targeted)", 0.0, 1.0, 0.10)
    with s2:
        uplift_pct = st.slider("Spend uplift for responders (%)", 0, 200, 20) / 100.0
    with s3:
        discount_pct = st.slider("Discount cost (% of incremental revenue)", 0, 80, 10) / 100.0
    with s4:
        fixed_campaign_cost = st.number_input("Fixed campaign cost (£)", min_value=0.0, value=0.0, step=50.0)

    roi_tbl = profit_tbl.copy()
    roi_tbl["responders"] = (roi_tbl["targeted_customers"] * response_rate).round().astype(int)
    roi_tbl["incr_rev"] = roi_tbl["responders"] * (roi_tbl["avg_revenue"] * uplift_pct)
    roi_tbl["discount_cost"] = roi_tbl["incr_rev"] * discount_pct
    roi_tbl["incr_gross_profit"] = roi_tbl["incr_rev"] * gross_margin
    roi_tbl["incr_net_profit"] = roi_tbl["incr_gross_profit"] - roi_tbl["discount_cost"] - roi_tbl["campaign_cost"]

    total_responders = int(roi_tbl["responders"].sum())
    if total_responders > 0:
        roi_tbl["fixed_cost_alloc"] = fixed_campaign_cost * (roi_tbl["responders"] / total_responders)
    else:
        roi_tbl["fixed_cost_alloc"] = fixed_campaign_cost / max(len(roi_tbl), 1)

    roi_tbl["incr_net_profit_after_fixed"] = roi_tbl["incr_net_profit"] - roi_tbl["fixed_cost_alloc"]
    roi_tbl["total_cost"] = roi_tbl["campaign_cost"] + roi_tbl["discount_cost"] + roi_tbl["fixed_cost_alloc"]
    roi_tbl["roi"] = np.where(roi_tbl["total_cost"] > 0, roi_tbl["incr_net_profit_after_fixed"] / roi_tbl["total_cost"], np.nan)

    st.markdown("### ✅ ROI results by cluster")
    roi_show = roi_tbl[[
        "cluster", "cluster_name", "targeted_customers", "responders",
        "incr_rev", "discount_cost", "campaign_cost", "fixed_cost_alloc",
        "incr_net_profit_after_fixed", "roi"
    ]].copy()

    st.dataframe(
        roi_show.round({
            "incr_rev": 0,
            "discount_cost": 0,
            "campaign_cost": 0,
            "fixed_cost_alloc": 0,
            "incr_net_profit_after_fixed": 0,
            "roi": 2
        }),
        use_container_width=True
    )

    total_incr_rev = float(roi_tbl["incr_rev"].sum())
    total_cost = float(roi_tbl["total_cost"].sum())
    total_profit = float(roi_tbl["incr_net_profit_after_fixed"].sum())
    overall_roi = (total_profit / total_cost) if total_cost > 0 else None

    st.markdown("### 🧾 Campaign summary")
    cA, cB, cC, cD = st.columns(4)
    cA.metric("Incremental revenue", f"£ {total_incr_rev:,.0f}")
    cB.metric("Total cost", f"£ {total_cost:,.0f}")
    cC.metric("Incremental profit", f"£ {total_profit:,.0f}")
    cD.metric("Overall ROI", "N/A" if overall_roi is None else f"{overall_roi:.2f}x")

    st.markdown("### 📥 Downloads")
    download_csv_button(rfm_out, "⬇️ Download RFM + Clustered Customers (CSV)", "rfm_clusters.csv")
    download_csv_button(profit_tbl, "⬇️ Download profitability table (CSV)", "cluster_profitability.csv")
    download_csv_button(roi_tbl, "⬇️ Download ROI simulation table (CSV)", "cluster_roi_simulation.csv")


# =========================================================
# ASSOCIATION RULES: MARKET BASKET
# =========================================================
else:
    st.markdown("## 🧺 Association Rules (Market Basket)")
    st.write("Upload Online Retail dataset. The app builds baskets by **InvoiceNo** and generates association rules.")

    with st.sidebar:
        st.markdown("### 📂 Upload Dataset")
        up = st.file_uploader("Upload Online Retail CSV/XLSX", type=["csv", "xlsx", "xls"], key="mba_upload")

    if up is None:
        st.info("Upload the Online Retail dataset to begin.")
        st.stop()

    df = read_file(up)

    required = ["InvoiceNo", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}. Please upload the correct Online Retail dataset.")
        st.stop()

    # Clean
    df = df.dropna(subset=["InvoiceNo", "Description"])
    df["Description"] = df["Description"].astype(str).str.strip()

    # Remove cancelled invoices
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

    st.markdown("### ✅ Basket Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Rows", f"{len(df):,}")
    s2.metric("Invoices (Baskets)", f"{len(baskets):,}")
    s3.metric("Unique Items", f"{df['Description'].nunique():,}")

    # One-hot encoding
    te = TransactionEncoder()
    arr = te.fit(baskets).transform(baskets)
    onehot = pd.DataFrame(arr, columns=te.columns_)

    # Frequent itemsets
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

    # Rules
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

    st.markdown("### 🧠 Top Drivers (Feature-Importance Style)")
    tmp = rules_disp.copy()
    tmp["driver"] = tmp["antecedents"].apply(lambda s: s.split(",")[0].strip())
    driver_strength = tmp.groupby("driver")[["lift", "confidence"]].mean().sort_values("lift", ascending=False).head(10)

    st.dataframe(driver_strength.round(3), use_container_width=True)

    fig, ax = plt.subplots()
    ax.bar(driver_strength.index.astype(str), driver_strength["lift"].values)
    ax.set_title("Top Drivers by Avg Lift")
    ax.set_xlabel("Driver Item")
    ax.set_ylabel("Avg Lift")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig, clear_figure=True)

    st.markdown("### 📥 Download")
    download_csv_button(rules_disp, "⬇️ Download Association Rules (CSV)", "association_rules.csv")
