import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt

from pathlib import Path
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# =========================================================
# PAGE CONFIG + PRODUCTIZED HEADER
# =========================================================
st.set_page_config(page_title="Retail Behavior Insights", page_icon="🛒", layout="wide")
st.title("🛒 Retail Behavior Insights Dashboard")

st.markdown(
    """
**Problem this app solves (real-world):**  
Retail teams need to understand **who** their customers are, **what** they buy together, and **when** they return — to run smarter campaigns (win-back, loyalty, bundles) and improve revenue.

**What you can do here:**  
- Segment customers into actionable groups (RFM + clustering)  
- Discover product bundles (affinity / market basket)  
- Track retention over time (cohort analysis)  
- Explore trends (revenue/orders over time)  
- Generate simple campaign recommendations + export target lists
"""
)


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


@st.cache_data(show_spinner=False)
def load_demo_dataset(path: Path) -> pd.DataFrame:
    return pd.read_excel(path)


def load_dataset():
    st.sidebar.markdown("## 📂 Dataset")

    use_demo = st.sidebar.checkbox("Use demo dataset (recommended)", value=True)

    allow_upload = st.sidebar.checkbox("Allow upload (optional)", value=False)
    uploaded = None
    if allow_upload and not use_demo:
        uploaded = st.sidebar.file_uploader(
            "Upload your own CSV/XLSX",
            type=["csv", "xlsx", "xls"]
        )

    if use_demo:
        if not DATA_PATH.exists():
            st.error("❌ Demo dataset not found. Upload it to: `data/Online Retail.xlsx`")
            st.stop()
        df = load_demo_dataset(DATA_PATH)
        st.sidebar.success("✅ Loaded demo dataset")
        return df

    if not allow_upload or uploaded is None:
        st.info("Enable demo dataset OR enable upload and upload a file.")
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


def segment_name(row) -> str:
    """
    Human-friendly names based on RFM characteristics.
    (Simple rules — good enough for a portfolio app.)
    """
    r, f, m = row["Recency"], row["Frequency"], row["Monetary"]
    if f >= 10 and m >= 1500 and r <= 30:
        return "High-Value Loyalists"
    if f >= 6 and r <= 60:
        return "Active Repeat Buyers"
    if r > 180 and f <= 2:
        return "At-Risk / Dormant"
    if f <= 2 and r <= 60:
        return "New / Early Stage"
    if m < 300 and f >= 4:
        return "Bargain / Low-Spend Repeat"
    return "Regular Customers"


# =========================================================
# LOAD + CLEAN DATA ONCE
# =========================================================
df_raw = load_dataset()

required = ["InvoiceNo", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID"]
missing = [c for c in required if c not in df_raw.columns]
if missing:
    st.error(f"Missing columns: {missing}. Upload the standard Online Retail dataset format.")
    st.stop()

df = df_raw.copy()
df = df.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "Description"])
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
df = df.dropna(subset=["InvoiceDate"])
df["CustomerID"] = df["CustomerID"].astype(int)

# remove cancellations + invalid rows
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[(df["Quantity"].astype(float) > 0) & (df["UnitPrice"].astype(float) > 0)]
df["Revenue"] = df["Quantity"].astype(float) * df["UnitPrice"].astype(float)
df["Description"] = df["Description"].astype(str).str.strip()

# Date helpers
df["Date"] = df["InvoiceDate"].dt.date
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
df["Week"] = df["InvoiceDate"].dt.to_period("W").astype(str)


# =========================================================
# NAVIGATION (PRODUCTIZED)
# =========================================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Choose module",
        ["🏠 Overview", "👥 Segments", "🧺 Product Affinity", "📆 Cohorts (Retention)", "📈 Trends", "🎯 Recommendations"],
        label_visibility="collapsed"
    )
    st.divider()

# Global filters (apply on most pages)
with st.sidebar:
    st.markdown("### 🔎 Global Filters")
    min_date = df["InvoiceDate"].min().date()
    max_date = df["InvoiceDate"].max().date()
    date_range = st.date_input("Date range", [min_date, max_date])

start = pd.to_datetime(date_range[0])
end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
df_f = df[(df["InvoiceDate"] >= start) & (df["InvoiceDate"] < end)].copy()

if df_f.empty:
    st.warning("No data in selected date range.")
    st.stop()

# KPIs for overview/top
total_revenue = df_f["Revenue"].sum()
customers = df_f["CustomerID"].nunique()
invoices = df_f["InvoiceNo"].nunique()
items = df_f["Description"].nunique()


# =========================================================
# SEGMENT COMPUTATION (RFM + clustering) stored in session
# =========================================================
def compute_rfm(df_in: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df_in["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df_in.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum")
    ).reset_index()
    return rfm


def run_clustering(rfm: pd.DataFrame, algo: str, params: dict):
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

    out = rfm.copy()
    out["cluster"] = labels
    out["segment_name"] = out.apply(segment_name, axis=1)

    sil, dbi = safe_cluster_metrics(X_scaled, labels)
    return out, X_scaled, sil, dbi


# =========================================================
# PAGE: OVERVIEW
# =========================================================
if page == "🏠 Overview":
    st.markdown("## 🏠 Overview")

    k1, k2, k3, k4 = st.columns(4)
    with k1: kpi_card("Total Revenue", f"£ {total_revenue:,.0f}")
    with k2: kpi_card("Customers", f"{customers:,}")
    with k3: kpi_card("Invoices", f"{invoices:,}")
    with k4: kpi_card("Unique Items", f"{items:,}")

    st.markdown("### What insights can you extract?")
    st.markdown(
        """
- **Segments:** Identify high-value vs at-risk customers  
- **Affinity:** Find product bundles customers buy together  
- **Cohorts:** See if customers return after first purchase  
- **Trends:** Spot weekly/monthly spikes in demand
"""
    )

    st.markdown("### Data preview")
    st.dataframe(df_f.head(15), use_container_width=True)

    st.markdown("### Revenue over time")
    ts = df_f.groupby("Week")["Revenue"].sum().reset_index()
    chart = alt.Chart(ts).mark_line(point=True).encode(
        x=alt.X("Week:N", title="Week"),
        y=alt.Y("Revenue:Q", title="Revenue"),
        tooltip=["Week", "Revenue"]
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)


# =========================================================
# PAGE: SEGMENTS
# =========================================================
elif page == "👥 Segments":
    st.markdown("## 👥 Customer Segments (RFM + Clustering)")
    st.caption("Goal: Identify customer groups to target with different campaigns (loyalty, win-back, onboarding).")

    rfm = compute_rfm(df_f)

    with st.sidebar:
        st.markdown("### 🧠 Segmentation Settings")
        algo = st.selectbox("Algorithm", ["K-Means", "Hierarchical", "GMM", "DBSCAN"])
        params = {}
        if algo in ["K-Means", "Hierarchical", "GMM"]:
            params["k"] = st.slider("Clusters", 2, 12, 4)
        else:
            params["eps"] = st.slider("eps", 0.1, 5.0, 0.6)
            params["min_samples"] = st.slider("min_samples", 2, 40, 5)

        run = st.button("🚀 Compute Segments", use_container_width=True)

    if not run and "rfm_out" not in st.session_state:
        st.info("Choose settings in sidebar and click **Compute Segments**.")
        st.stop()

    if run:
        rfm_out, X_scaled, sil, dbi = run_clustering(rfm, algo, params)
        st.session_state["rfm_out"] = rfm_out
        st.session_state["X_scaled"] = X_scaled
        st.session_state["metrics"] = {"algo": algo, "sil": sil, "dbi": dbi}

    rfm_out = st.session_state["rfm_out"]
    metrics = st.session_state.get("metrics", {})
    st.markdown("### ✅ Segment quality")
    st.write(f"**Algorithm:** {metrics.get('algo', 'N/A')}")
    st.write(f"**Silhouette:** {metrics.get('sil', 'N/A')}")
    st.write(f"**Davies–Bouldin:** {metrics.get('dbi', 'N/A')}")

    st.markdown("### Segment distribution")
    seg_counts = rfm_out["segment_name"].value_counts().reset_index()
    seg_counts.columns = ["segment_name", "customers"]

    bar = alt.Chart(seg_counts).mark_bar().encode(
        x=alt.X("segment_name:N", sort="-y", title="Segment"),
        y=alt.Y("customers:Q", title="Customers"),
        tooltip=["segment_name", "customers"]
    ).properties(height=320)
    st.altair_chart(bar, use_container_width=True)

    st.markdown("### Segment metrics (behavior insights)")
    seg_metrics = rfm_out.groupby("segment_name").agg(
        Customers=("CustomerID", "nunique"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
    ).reset_index()

    st.dataframe(seg_metrics.sort_values("Customers", ascending=False), use_container_width=True)

    st.markdown("### 🗺️ PCA Scatter Plot (clusters)")
    plot_pca_scatter(st.session_state["X_scaled"], rfm_out["cluster"].values, f"{metrics.get('algo','')} — RFM Clusters (PCA)")

    st.markdown("### 📥 Export target list")
    download_csv_button(rfm_out.sort_values("Monetary", ascending=False), "⬇️ Download Customer Segments (CSV)", "customer_segments.csv")


# =========================================================
# PAGE: PRODUCT AFFINITY
# =========================================================
elif page == "🧺 Product Affinity":
    st.markdown("## 🧺 Product Affinity (Bundles / Cross-sell)")
    st.caption("Goal: Identify products frequently purchased together to create bundles and recommendations.")

    # Basket preparation
    baskets = df_f.groupby("InvoiceNo")["Description"].apply(list).tolist()

    with st.sidebar:
        st.markdown("### ⚙️ Affinity Settings")
        algo = st.selectbox("Algorithm", ["Apriori", "FP-Growth"])
        min_support = st.slider("min_support", 0.001, 0.2, 0.02)
        min_conf = st.slider("min_confidence", 0.01, 1.0, 0.2)
        min_lift = st.slider("min_lift", 0.5, 5.0, 1.0)
        run = st.button("🚀 Generate Affinity", use_container_width=True)

    if not run:
        st.info("Set parameters in sidebar and click **Generate Affinity**.")
        st.stop()

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

    st.markdown("### 🔥 Top association rules")
    st.dataframe(rules_disp.head(30), use_container_width=True)

    st.markdown("### 📊 Top bundles (by lift)")
    top_pairs = rules_disp.head(15).copy()
    top_pairs["rule"] = top_pairs["antecedents"] + " → " + top_pairs["consequents"]

    chart = alt.Chart(top_pairs).mark_bar().encode(
        x=alt.X("lift:Q", title="Lift"),
        y=alt.Y("rule:N", sort="-x", title="Bundle rule"),
        tooltip=["support", "confidence", "lift"]
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)

    # Optional network graph
    st.markdown("### 🕸️ Product network (optional)")
    try:
        import networkx as nx

        G = nx.Graph()
        for _, r in top_pairs.iterrows():
            a = r["antecedents"]
            c = r["consequents"]
            lift = float(r["lift"])
            # create edges between each antecedent item and each consequent item
            for ai in [x.strip() for x in a.split(",")]:
                for ci in [x.strip() for x in c.split(",")]:
                    if ai and ci and ai != ci:
                        G.add_edge(ai, ci, weight=lift)

        fig, ax = plt.subplots(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=7)
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx(G, pos=pos, ax=ax, with_labels=True, node_size=600, font_size=8, width=weights)
        ax.set_axis_off()
        st.pyplot(fig, clear_figure=True)
    except Exception:
        st.info("Install `networkx` to enable the product network graph. (The bundles chart above already gives the insight.)")

    st.markdown("### 📥 Download rules")
    download_csv_button(rules_disp, "⬇️ Download Rules (CSV)", "association_rules.csv")


# =========================================================
# PAGE: COHORT RETENTION
# =========================================================
elif page == "📆 Cohorts (Retention)":
    st.markdown("## 📆 Cohort Analysis (Retention)")
    st.caption("Goal: Measure how often customers return after their first purchase (retention).")

    d = df_f[["CustomerID", "InvoiceDate", "InvoiceNo"]].copy()
    d["OrderMonth"] = d["InvoiceDate"].dt.to_period("M").dt.to_timestamp()

    # Cohort = first purchase month per customer
    first_purchase = d.groupby("CustomerID")["OrderMonth"].min().reset_index()
    first_purchase.columns = ["CustomerID", "CohortMonth"]
    d = d.merge(first_purchase, on="CustomerID", how="left")

    # Cohort index (months since first purchase)
    d["CohortIndex"] = (
        (d["OrderMonth"].dt.year - d["CohortMonth"].dt.year) * 12
        + (d["OrderMonth"].dt.month - d["CohortMonth"].dt.month)
        + 1
    )

    cohort_counts = d.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().reset_index()
    cohort_pivot = cohort_counts.pivot(index="CohortMonth", columns="CohortIndex", values="CustomerID").fillna(0)

    # retention %
    cohort_sizes = cohort_pivot[1]
    retention = cohort_pivot.divide(cohort_sizes, axis=0).round(3)

    st.markdown("### Retention heatmap")
    heat_df = retention.reset_index().melt(id_vars="CohortMonth", var_name="Month", value_name="Retention")
    heat_df["CohortMonth"] = heat_df["CohortMonth"].dt.strftime("%Y-%m")

    heat = alt.Chart(heat_df).mark_rect().encode(
        x=alt.X("Month:O", title="Months since first purchase"),
        y=alt.Y("CohortMonth:N", title="Cohort month"),
        color=alt.Color("Retention:Q"),
        tooltip=["CohortMonth", "Month", "Retention"]
    ).properties(height=420)
    st.altair_chart(heat, use_container_width=True)

    st.markdown("### Retention table (fraction)")
    st.dataframe(retention, use_container_width=True)

    st.markdown("### 📥 Export retention table")
    download_csv_button(retention.reset_index(), "⬇️ Download Cohort Retention (CSV)", "cohort_retention.csv")


# =========================================================
# PAGE: TRENDS
# =========================================================
elif page == "📈 Trends":
    st.markdown("## 📈 Trends (Revenue + Orders over time)")
    st.caption("Goal: Identify seasonality/spikes and understand demand patterns.")

    granularity = st.selectbox("Time granularity", ["Week", "Month"], index=0)

    if granularity == "Week":
        grp = df_f.groupby("Week").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
        xcol = "Week"
    else:
        grp = df_f.groupby("Month").agg(Revenue=("Revenue", "sum"), Orders=("InvoiceNo", "nunique")).reset_index()
        xcol = "Month"

    st.markdown("### Revenue over time")
    rev = alt.Chart(grp).mark_line(point=True).encode(
        x=alt.X(f"{xcol}:N", title=granularity),
        y=alt.Y("Revenue:Q", title="Revenue"),
        tooltip=[xcol, "Revenue", "Orders"]
    ).properties(height=320)
    st.altair_chart(rev, use_container_width=True)

    st.markdown("### Orders over time")
    ords = alt.Chart(grp).mark_line(point=True).encode(
        x=alt.X(f"{xcol}:N", title=granularity),
        y=alt.Y("Orders:Q", title="Orders"),
        tooltip=[xcol, "Revenue", "Orders"]
    ).properties(height=320)
    st.altair_chart(ords, use_container_width=True)

    st.markdown("### Top products (by revenue)")
    top_products = df_f.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(15).reset_index()
    bar = alt.Chart(top_products).mark_bar().encode(
        x=alt.X("Revenue:Q", title="Revenue"),
        y=alt.Y("Description:N", sort="-x", title="Product"),
        tooltip=["Description", "Revenue"]
    ).properties(height=420)
    st.altair_chart(bar, use_container_width=True)


# =========================================================
# PAGE: RECOMMENDATIONS
# =========================================================
else:
    st.markdown("## 🎯 Recommendations (Campaign Ideas + Target Lists)")
    st.caption("Goal: Convert analytics into actions (who to target and what to do next).")

    rfm = compute_rfm(df_f)

    # If user hasn’t computed segments yet, do a default quick segmentation
    if "rfm_out" not in st.session_state:
        default_algo = "K-Means"
        default_params = {"k": 4}
        rfm_out, X_scaled, sil, dbi = run_clustering(rfm, default_algo, default_params)
        st.session_state["rfm_out"] = rfm_out
        st.session_state["X_scaled"] = X_scaled
        st.session_state["metrics"] = {"algo": default_algo, "sil": sil, "dbi": dbi}

    rfm_out = st.session_state["rfm_out"]

    segs = sorted(rfm_out["segment_name"].unique().tolist())
    chosen = st.selectbox("Choose a segment", segs)

    seg_df = rfm_out[rfm_out["segment_name"] == chosen].copy()
    st.markdown("### Segment summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{seg_df['CustomerID'].nunique():,}")
    c2.metric("Avg Frequency", f"{seg_df['Frequency'].mean():.2f}")
    c3.metric("Avg Monetary", f"£ {seg_df['Monetary'].mean():.0f}")

    st.markdown("### Suggested actions")
    actions = {
        "High-Value Loyalists": [
            "Offer loyalty rewards / early access",
            "Avoid heavy discounts; focus on retention perks",
            "Recommend premium bundles"
        ],
        "Active Repeat Buyers": [
            "Personalized recommendations based on recent purchases",
            "Bundles / cross-sell offers",
            "Free shipping threshold incentives"
        ],
        "New / Early Stage": [
            "Onboarding email series",
            "First-to-second purchase incentive",
            "Recommend best-sellers"
        ],
        "At-Risk / Dormant": [
            "Win-back campaign (limited-time offer)",
            "Reminder emails and personalized discounts",
            "Target based on their last purchased category"
        ],
        "Bargain / Low-Spend Repeat": [
            "Deal alerts and seasonal promotions",
            "Bundle smaller items",
            "Low-cost add-ons at checkout"
        ],
        "Regular Customers": [
            "Nudge toward repeat purchase",
            "Cross-sell with top bundles",
            "Seasonal category recommendations"
        ]
    }
    for a in actions.get(chosen, ["Target with tailored offers based on prior purchases."]):
        st.write(f"- {a}")

    st.markdown("### Export target list")
    download_csv_button(seg_df.sort_values(["Monetary", "Frequency"], ascending=False),
                        "⬇️ Download Target Customers (CSV)",
                        f"targets_{chosen.lower().replace(' ','_').replace('/','_')}.csv")

    st.markdown("### Join with transactions to see top products for this segment")
    seg_customers = set(seg_df["CustomerID"].tolist())
    seg_tx = df_f[df_f["CustomerID"].isin(seg_customers)]

    top_seg_products = seg_tx.groupby("Description")["Revenue"].sum().sort_values(ascending=False).head(15).reset_index()
    chart = alt.Chart(top_seg_products).mark_bar().encode(
        x=alt.X("Revenue:Q", title="Revenue"),
        y=alt.Y("Description:N", sort="-x", title="Top products"),
        tooltip=["Description", "Revenue"]
    ).properties(height=420)
    st.altair_chart(chart, use_container_width=True)
