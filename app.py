# chainforecast_final.py
# ChainForecast — Final working app
# - Uses uploaded file or fallback to /mnt/data/cleaned_online_retail.xlsx
# - Fixes float customerid issue (13085.0 -> "13085")
# - 4 tabs: Forecasting, Customer Analytics, Top Products, Product Boom
# - SARIMAX (short-term) & XGBoost (long-term)
# - RFM & CRM coupon assignment
# - Aurora-style UI (static)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import hashlib
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import os

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="ChainForecast", layout="wide")

# --------------------------------------------------
# CSS — Aurora-ish UI
# --------------------------------------------------
AURORA_CSS = """
<style>
body { 
    background: radial-gradient(ellipse at top left, #051829 0%, #000914 55%, #000000 100%) !important;
    color: #e8f0ff !important;
    font-family: 'Inter', sans-serif !important;
}
.navbar {
    position: fixed;
    top: 10px;
    left: 20px;
    right: 20px;
    height: 64px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    padding: 10px 22px;
    background: rgba(6,15,30,0.55);
    backdrop-filter: blur(14px);
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
    z-index: 9999;
}
.brand { font-size: 1.3rem; font-weight: 800; color: #ffffff; }
.main-wrap { margin-top: 90px; padding-left: 25px; padding-right: 25px; }
.dashboard-box {
    padding:20px;
    border-radius:16px;
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.12);
    backdrop-filter:blur(18px);
    box-shadow:0 6px 25px rgba(0,0,0,0.5);
    margin-bottom:25px;
}
.card-meta { color: #aac0d0; font-size: 0.9rem; margin-bottom: 10px; }
</style>
"""
st.markdown(AURORA_CSS, unsafe_allow_html=True)

# --------------------------------------------------
# Navbar (static)
# --------------------------------------------------
st.markdown("""
<div class='navbar'>
    <div class='brand'>ChainForecast Dashboard</div>
    <div style="display:flex;align-items:center;gap:10px;">
        <div style="text-align:right;">
            <div style="font-weight:600;">Retailer Admin</div>
            <div style="font-size:0.8rem;color:#9db2c6;">Profile</div>
        </div>
        <div style="width:40px;height:40px;border-radius:50%;
             background:linear-gradient(135deg,#3b82f6,#ec4899);
             display:flex;align-items:center;justify-content:center;
             color:white;font-weight:700;">
             RA
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div class='main-wrap'>", unsafe_allow_html=True)

# --------------------------------------------------
# Helpers: forecasting, rfm, etc.
# --------------------------------------------------


def prepare_weekly_series(d):
    d = d.copy()
    d['invoicedate'] = pd.to_datetime(d['invoicedate'], errors='coerce')
    d = d.set_index('invoicedate').sort_index()
    weekly = d['sales'].resample(
        'W-MON').sum().reset_index().rename(columns={'invoicedate': 'ds', 'sales': 'y'})
    weekly['ds'] = pd.to_datetime(weekly['ds']).dt.to_period(
        'W').apply(lambda p: p.start_time)
    return weekly


def train_sarimax(weekly):
    model = SARIMAX(weekly['y'], order=(1, 1, 1), seasonal_order=(0, 1, 1, 5),
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.fit(disp=False)
    return res


def forecast_sarimax(res, steps=4):
    pred = res.get_forecast(steps=steps)
    return pred.predicted_mean.values


def create_lag_features_weekly(weekly, lags=[1, 2, 3, 4, 6, 8, 12]):
    df_ = weekly.copy()
    for lag in lags:
        df_[f'lag_{lag}'] = df_['y'].shift(lag)
    return df_.dropna().reset_index(drop=True)


def train_xgb(weekly):
    df_ = create_lag_features_weekly(weekly)
    feat_cols = [c for c in df_.columns if c.startswith('lag_')]
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(df_[feat_cols], df_['y'])
    return model, feat_cols


def forecast_xgb(model, weekly, feat_cols, steps=4):
    history = weekly['y'].tolist()
    preds = []
    for _ in range(steps):
        fv = []
        for f in feat_cols:
            lag = int(f.split('_')[1])
            fv.append(history[-lag] if lag <=
                      len(history) else np.mean(history))
        p = model.predict(np.array(fv).reshape(1, -1))[0]
        preds.append(float(p))
        history.append(p)
    return preds


def rfm_analysis(df):
    d = df.copy()
    d['invoicedate'] = pd.to_datetime(d['invoicedate'])
    snapshot = d['invoicedate'].max() + pd.Timedelta(days=1)
    rfm = d.groupby('customerid').agg({
        'invoicedate': lambda x: (snapshot - x.max()).days,
        'customerid': 'count',
        'sales': 'sum'
    }).rename(columns={'invoicedate': 'recency', 'customerid': 'frequency', 'sales': 'monetary'})
    rfm['recency_score'] = pd.qcut(rfm['recency'], 4, labels=[
                                   4, 3, 2, 1]).astype(int)
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(
        method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['monetary_score'] = pd.qcut(
        rfm['monetary'], 4, labels=[1, 2, 3, 4]).astype(int)
    rfm['rfm_score'] = rfm['recency_score']*100 + \
        rfm['frequency_score']*10 + rfm['monetary_score']
    return rfm.reset_index()


def segmentation_kmeans(rfm_df, n_clusters=4):
    X = rfm_df[['recency', 'frequency', 'monetary']]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42).fit(Xs)
    rfm_df['segment'] = km.labels_
    order = rfm_df.groupby('segment')['monetary'].mean(
    ).sort_values(ascending=False).index
    mapping = {seg: f"Segment_{i+1}" for i, seg in enumerate(order)}
    rfm_df['segment_label'] = rfm_df['segment'].map(mapping)
    return rfm_df


def top_products_last_n_days(df, days=60):
    cutoff = pd.to_datetime(df['invoicedate']).max() - pd.Timedelta(days=days)
    recent = df[pd.to_datetime(df['invoicedate']) >= cutoff]
    key = 'stockcode' if 'stockcode' in recent.columns else 'description'
    return recent.groupby(key)['sales'].sum().reset_index().sort_values('sales', ascending=False)


def retention_rate(df):
    first = df.groupby('customerid')['invoicedate'].min(
    ).reset_index().rename(columns={'invoicedate': 'first_date'})
    merged = df.merge(first, on='customerid', how='left')
    merged['is_repeat'] = pd.to_datetime(
        merged['invoicedate']) > pd.to_datetime(merged['first_date'])
    return merged.groupby('customerid')['is_repeat'].any().mean()


def compute_hash_bytes(b: bytes):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def merkle_root_from_ids(ids):
    if len(ids) == 0:
        return None
    leaves = [hashlib.sha256(str(i).encode()).digest() for i in ids]
    while len(leaves) > 1:
        if len(leaves) % 2 == 1:
            leaves.append(leaves[-1])
        new_level = []
        for i in range(0, len(leaves), 2):
            new_level.append(hashlib.sha256(leaves[i] + leaves[i+1]).digest())
        leaves = new_level
    return hashlib.sha256(leaves[0]).hexdigest()


# --------------------------------------------------
# Load dataset — uploaded or fallback local path
# --------------------------------------------------
DEFAULT_PATH = "/mnt/data/cleaned_online_retail.xlsx"  # <-- your uploaded file path

uploaded = st.file_uploader(
    "Upload dataset (.csv or .xlsx). If none, app will try the demo file at /mnt/data/cleaned_online_retail.xlsx", type=['csv', 'xlsx'])

raw_bytes = None
if uploaded is not None:
    try:
        if uploaded.name.lower().endswith('.csv'):
            raw = pd.read_csv(uploaded, encoding='latin1', low_memory=False)
            raw_bytes = uploaded.getvalue()
        else:
            raw = pd.read_excel(uploaded, engine='openpyxl')
            raw_bytes = uploaded.getvalue()
        st.success("File uploaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
else:
    if os.path.exists(DEFAULT_PATH):
        try:
            raw = pd.read_excel(DEFAULT_PATH, engine='openpyxl')
            with open(DEFAULT_PATH, 'rb') as f:
                raw_bytes = f.read()
            st.info(f"Loaded demo dataset from {DEFAULT_PATH}")
        except Exception as e:
            st.error(f"Failed to read demo file: {e}")
            st.stop()
    else:
        st.info("Please upload a dataset (.csv or .xlsx) or put the demo file at /mnt/data/cleaned_online_retail.xlsx")
        st.stop()

# --------------------------------------------------
# Pre-clean fixes
#  - customerid float fix: '13085.0' -> '13085'
# --------------------------------------------------
if 'customerid' in raw.columns:
    # Convert to string and strip trailing .0 if present
    raw['customerid'] = raw['customerid'].astype(
        str).str.replace('.0$', '', regex=True).str.strip()

# If invoice date column named slightly different, try to find it (but in your dataset it's 'invoicedate')
# We'll trust 'invoicedate' exists per your sample.

# --------------------------------------------------
# Cleaning (light)
# --------------------------------------------------


def clean_record(df):
    df = df.copy()
    # normalize column names
    df.columns = [c.replace(" ", "").replace("_", "").lower()
                  for c in df.columns]
    rename = {
        'invoice': 'invoiceno',
        'invoiceno': 'invoiceno',
        'stockcode': 'stockcode',
        'description': 'description',
        'quantity': 'quantity',
        'invoicedate': 'invoicedate',
        'price': 'price',
        'customerid': 'customerid',
        'country': 'country',
        'totalprice': 'totalprice'
    }
    df = df.rename(columns=rename)
    # remove possible credits
    if 'invoiceno' in df.columns:
        df = df[~df['invoiceno'].astype(str).str.startswith("C")]
    # invoice date
    if 'invoicedate' in df.columns:
        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        df = df.dropna(subset=['invoicedate'])
    # customerid
    if 'customerid' in df.columns:
        df['customerid'] = df['customerid'].astype(str).str.strip()
    else:
        df['customerid'] = df.index.astype(str)
    # numeric cleanup
    if 'quantity' in df.columns and 'price' in df.columns:
        df['quantity'] = pd.to_numeric(
            df['quantity'], errors='coerce').fillna(0)
        df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0)
        df = df[(df['quantity'] > 0) & (df['price'] > 0)]
    # description
    if 'description' in df.columns:
        df['description'] = df['description'].astype(str).str.strip()
    # sales
    if 'totalprice' in df.columns:
        df['sales'] = pd.to_numeric(
            df['totalprice'], errors='coerce').fillna(0.0)
    else:
        df['sales'] = df['quantity'] * df['price']
    return df.reset_index(drop=True)


with st.spinner("Cleaning data..."):
    df = clean_record(raw)

# compute file hash & merkle
file_hash = compute_hash_bytes(raw_bytes) if raw_bytes is not None else None
merkle = merkle_root_from_ids(df.index.tolist())

# basic validation
if 'invoicedate' not in df.columns:
    st.error("Required column 'invoicedate' missing after cleaning.")
    st.stop()
if 'customerid' not in df.columns:
    st.error("Required column 'customerid' missing after cleaning.")
    st.stop()

# --------------------------------------------------
# KPI row
# --------------------------------------------------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"₹{df['sales'].sum():,.0f}")
col2.metric("Unique Customers", df['customerid'].nunique())
col3.metric("Orders", df['invoiceno'].nunique()
            if 'invoiceno' in df.columns else len(df))
col4.metric("Repeat Rate", f"{retention_rate(df)*100:.1f}%")

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_forecast, tab_customer, tab_top, tab_boom = st.tabs(
    ["Forecasting", "Customer Analytics", "Top Products & Retention", "Product Boom"])

# -------------------------
# TAB: Forecasting
# -------------------------
with tab_forecast:
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.header("Forecasting — Product-level")
    st.markdown("<div class='card-meta'>Enter product ID (stockcode) or description substring. Train SARIMAX (short-term) & XGBoost (long-term).</div>", unsafe_allow_html=True)

    product_input = st.text_input(
        "Product ID (stockcode) or description substring", value="", key="prod_input")
    if product_input:
        mask = (df.get('stockcode', '').astype(str) == product_input) | df['description'].str.contains(
            product_input, case=False, na=False)
        prod_df = df[mask]
        if prod_df.empty:
            st.warning("No product matches that input.")
        else:
            st.subheader(f"Product sample: {prod_df['description'].iloc[0]}")
            weekly = prepare_weekly_series(prod_df)
            st.plotly_chart(px.line(weekly, x='ds', y='y',
                            title='Weekly Sales'), use_container_width=True)
            if len(weekly) < 8:
                st.warning(
                    "Not enough weekly history (~8+ weeks recommended) to train models.")
            else:
                if st.button("Train SARIMAX + XGBoost for product"):
                    with st.spinner("Training models..."):
                        sar_res = train_sarimax(weekly)
                        sar_preds = forecast_sarimax(sar_res, steps=4)
                        xgb_model, xgb_feats = train_xgb(weekly)
                        xgb_preds = forecast_xgb(
                            xgb_model, weekly, xgb_feats, steps=4)
                        last_week = weekly['ds'].max()
                        future_weeks = [last_week +
                                        timedelta(weeks=i+1) for i in range(4)]
                        forecast_df = pd.DataFrame(
                            {'ds': future_weeks, 'SARIMAX': sar_preds, 'XGBoost': xgb_preds})
                        st.session_state['product_forecast'] = {
                            'product': product_input, 'forecast': forecast_df, 'weekly': weekly}
                    st.success("Models trained and stored in session.")
                if 'product_forecast' in st.session_state and st.session_state['product_forecast']['product'] == product_input:
                    fo = st.session_state['product_forecast']['forecast']
                    weekly = st.session_state['product_forecast']['weekly']
                    combined = pd.concat([weekly.rename(columns={'y': 'Actual'}).set_index(
                        'ds'), fo.set_index('ds')], axis=0).reset_index()
                    cols = [c for c in ['Actual', 'SARIMAX',
                                        'XGBoost'] if c in combined.columns]
                    st.plotly_chart(px.line(
                        combined, x='ds', y=cols, title='Historical + Forecast'), use_container_width=True)
                    st.dataframe(fo)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB: Customer Analytics
# -------------------------
with tab_customer:
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.header("Customer Analytics — RFM, Top Products, CRM")
    st.markdown("<div class='card-meta'>Enter Customer ID to see profile (recency, frequency, monetary), top 50 products, and assign coupon/discount for their segment.</div>", unsafe_allow_html=True)

    # Ensure customerid strings are normalized (this is the key fix)
    df['customerid'] = df['customerid'].astype(
        str).str.replace('.0$', '', regex=True).str.strip()

    cust_input = st.text_input("Customer ID", value="", key="cust_input")
    if cust_input:
        cid = str(cust_input).strip()
        cust_df = df[df['customerid'] == cid]
        if cust_df.empty:
            st.warning(
                "Customer not found. Make sure you entered the ID without decimals (e.g., 13085 not 13085.0).")
            # show a few closest candidates to help user
            try:
                unique_ids = df['customerid'].unique().astype(str)
                import difflib
                matches = difflib.get_close_matches(
                    cid, unique_ids, n=8, cutoff=0.6)
                if matches:
                    st.info("Close matches (select to view):")
                    sel = st.selectbox("Close matches", [
                                       "None"] + matches, key="close_matches")
                    if sel and sel != "None":
                        cust_df = df[df['customerid'] == sel]
                        cid = sel
                else:
                    st.info("No close matches found.")
            except Exception:
                pass
        if not cust_df.empty:
            st.success(f"Customer {cid} found — showing profile")

            # RFM & segmentation
            rfm = rfm_analysis(df)
            seg = segmentation_kmeans(rfm)
            cust_row = seg[seg['customerid'] == cid]
            if cust_row.empty:
                st.error("RFM/segmentation not found for this customer.")
            else:
                rec = int(cust_row['recency'].iloc[0])
                freq = int(cust_row['frequency'].iloc[0])
                mon = float(cust_row['monetary'].iloc[0])
                seg_label = cust_row['segment_label'].iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Recency (days)", rec)
                c2.metric("Frequency", freq)
                c3.metric("Monetary", f"₹{mon:,.0f}")
                c4.metric("Segment", seg_label)

                st.markdown("---")
                st.subheader("Top 50 Products Purchased by Customer")
                top50 = cust_df.groupby('stockcode').agg({'sales': 'sum', 'description': 'first', 'quantity': 'sum'}).reset_index(
                ).sort_values('sales', ascending=False).head(50)
                st.dataframe(top50)
                if not top50.empty:
                    st.plotly_chart(px.bar(top50.head(15), x='stockcode', y='sales', hover_data=[
                                    'description', 'quantity'], title=f"Top products for {cid}"), use_container_width=True)
                    st.download_button("Download Top 50 CSV", data=top50.to_csv(
                        index=False).encode('utf-8'), file_name=f"top50_customer_{cid}.csv")

                st.markdown("---")
                st.subheader(
                    "CRM: Assign Discount & Coupon for this customer's segment")
                if 'offers' not in st.session_state:
                    st.session_state['offers'] = {}
                if seg_label not in st.session_state['offers']:
                    st.session_state['offers'][seg_label] = {
                        'discount_pct': 10, 'coupon': f'{seg_label}_10OFF'}
                dcol, ccol = st.columns(2)
                with dcol:
                    disc = st.number_input(f"Discount % for {seg_label}", min_value=0, max_value=100,
                                           value=st.session_state['offers'][seg_label]['discount_pct'], key=f"disc_{seg_label}")
                with ccol:
                    coupon = st.text_input(
                        f"Coupon for {seg_label}", value=st.session_state['offers'][seg_label]['coupon'], key=f"coupon_{seg_label}")
                st.session_state['offers'][seg_label] = {
                    'discount_pct': int(disc), 'coupon': coupon}
                st.info(
                    f"Segment {seg_label} → Discount: {disc}%  Coupon: {coupon}")

    else:
        st.info("Enter Customer ID to begin (e.g., 13085)")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB: Top Products & Retention
# -------------------------
with tab_top:
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.header("Top Products & Retention")
    st.markdown("<div class='card-meta'>Snapshot of top products in last 60 days and repeat-purchase metrics.</div>", unsafe_allow_html=True)

    top_prods = top_products_last_n_days(df, days=60)
    st.subheader("Top products (last 60 days)")
    st.dataframe(top_prods.head(50))
    if not top_prods.empty:
        st.plotly_chart(px.bar(top_prods.head(
            10), x=top_prods.columns[0], y='sales', title="Top 10 products (60d)"), use_container_width=True)
        best = top_prods.iloc[0]
        st.success(
            f"Top product (60d): {best[top_prods.columns[0]]} — Sales: {best['sales']:.2f}")

    rr = retention_rate(df)
    st.metric("Repeat purchase rate", f"{rr*100:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# TAB: Product Boom Predictions
# -------------------------
with tab_boom:
    st.markdown("<div class='dashboard-box'>", unsafe_allow_html=True)
    st.header("Product Boom Prediction")
    st.markdown("<div class='card-meta'>Use XGBoost to forecast and rank growth candidates.</div>",
                unsafe_allow_html=True)

    try:
        candidates = df['stockcode'].astype(str).value_counts().index.tolist()
    except Exception:
        candidates = df['description'].astype(
            str).value_counts().index.tolist()

    sel = st.multiselect("Select products to forecast",
                         options=candidates, default=candidates[:10])
    horizon = st.number_input(
        "Forecast horizon (weeks)", min_value=1, max_value=52, value=4)
    if st.button("Run product boom predictions"):
        results = []
        with st.spinner("Forecasting..."):
            for p in sel:
                sub = df[(df.get('stockcode', '').astype(str)
                          == str(p)) | (df['description'] == p)]
                if len(sub) < 30:
                    continue
                weekly_p = prepare_weekly_series(sub)
                if len(weekly_p) < 8:
                    continue
                try:
                    model_p, feats_p = train_xgb(weekly_p)
                    preds_p = forecast_xgb(
                        model_p, weekly_p, feats_p, steps=horizon)
                    past = weekly_p['y'].tail(horizon).sum() if len(
                        weekly_p) >= horizon else weekly_p['y'].sum()
                    future = float(np.sum(preds_p))
                    growth = (future - past) / (past + 1e-9) * 100
                    results.append({'product': str(
                        p), 'past_sum': past, 'future_sum': future, 'growth_pct': growth})
                except Exception:
                    continue
        res_df = pd.DataFrame(results).sort_values(
            'growth_pct', ascending=False)
        st.dataframe(res_df.head(50))
        if not res_df.empty:
            topc = res_df.iloc[0]
            st.success(
                f"Top predicted boom product: {topc['product']} (+{topc['growth_pct']:.1f}%)")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Exports & integrity
# -------------------------
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("Exports & Data Integrity"):
    st.download_button("Download cleaned CSV", data=df.to_csv(index=False).encode(
        'utf-8'), file_name="cleaned_online_retail_cleaned.csv")
    if raw_bytes:
        st.write("SHA256 (uploaded/demo):")
        st.code(file_hash)
    if merkle:
        st.write("Merkle root (rows):")
        st.code(merkle)

st.markdown("</div>", unsafe_allow_html=True)

# End of file
