# AI Retail Analytics 🛒
Customer Segmentation + Market Basket Analysis using clustering and association rule mining.

## Methods
### Clustering
- K-Means
- Hierarchical (Agglomerative)
- Gaussian Mixture Models (GMM)
- DBSCAN

### Association Rule Mining
- Apriori
- FP-Growth  
(Produces rules with support, confidence, lift)

## Streamlit App
### Customer Segmentation tab
Upload a customer-level dataset (CSV/XLSX) with numeric features (e.g., RFM).
- Select features
- Run clustering
- View PCA plot
- See Silhouette + Davies–Bouldin scores
- Download clustered results

### Association Rules tab
Upload a transaction-level dataset (CSV/XLSX) with:
- Transaction ID column (e.g., InvoiceNo)
- Item column (e.g., Description)
Generate frequent itemsets + rules and download to CSV.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
