import os
import pandas as pd
import numpy as np
from datetime import timedelta
from IPython.display import display

# Base paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, "dataset")
input_dir = os.path.join(dataset_dir, "cleaned_online_retail.xlsx")
output_xl = os.path.join(dataset_dir, "cleaned_online_retail_rfm.xlsx")


def load_cleaned_csv(path=input_dir):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Excel file not found at: {path}")

    df = pd.read_excel(path, engine='openpyxl')

    # Normalize column names -> lowercase + no spaces
    df.columns = [str(c).strip().lower().replace(" ", "") for c in df.columns]

    # ---- Fix InvoiceDate ----
    if "invoicedate" not in df.columns:
        raise ValueError("Column 'invoicedate' missing.")
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

    # ---- Fix CustomerID ----
    if "customerid" not in df.columns:
        df["customerid"] = np.nan

    # ---- Fix Description ----
    if "description" not in df.columns:
        df["description"] = ""
    df["description"] = df["description"].astype(str).str.strip()

    # ---- Fix Quantity ----
    if "quantity" not in df.columns:
        df["quantity"] = np.nan
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # ---- Fix Price ----
    if "price" not in df.columns:
        df["price"] = np.nan
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    # ---- Fix TotalPrice (Monetary) ----
    if "totalprice" in df.columns:
        pass
    elif "sales" in df.columns:        # your file has "sales"
        df.rename(columns={"sales": "totalprice"}, inplace=True)
    else:
        df["totalprice"] = df["quantity"] * df["price"]

    # Remove invalid rows
    df = df[df["invoicedate"].notna() & df["totalprice"].notna()].copy()

    return df


def compute_rfm_and_assign(df):
    rfm_df = df[df["customerid"].notna()].copy()

    if rfm_df.empty:
        return pd.DataFrame(columns=[
            "customerid", "recency", "frequency", "monetary",
            "customertype", "discount", "r_score", "f_score",
            "m_score", "rfm_sum"
        ])

    snapshot = rfm_df["invoicedate"].max() + timedelta(days=1)

    # Group for RFM metrics
    if "invoiceno" in df.columns:
        grouped = rfm_df.groupby("customerid").agg({
            "invoicedate": lambda x: (snapshot - x.max()).days,
            "totalprice": "sum",
            "invoiceno": "nunique"
        })
        grouped.rename(columns={"invoiceno": "frequency"}, inplace=True)
    else:
        grouped = rfm_df.groupby("customerid").agg({
            "invoicedate": lambda x: (snapshot - x.max()).days,
            "totalprice": "sum"
        })
        grouped["frequency"] = rfm_df.groupby("customerid").size()

    grouped.rename(columns={
        "invoicedate": "recency",
        "totalprice": "monetary"
    }, inplace=True)

    grouped = grouped.reset_index()

    # Compute RFM scores
    grouped["r_score"] = pd.qcut(
        grouped["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]
    ).astype(int)

    grouped["f_score"] = pd.qcut(
        grouped["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]
    ).astype(int)

    grouped["m_score"] = pd.qcut(
        grouped["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]
    ).astype(int)

    grouped["rfm_sum"] = (
        grouped["r_score"] + grouped["f_score"] + grouped["m_score"]
    )

    # Assign customer segments
    def get_type(row):
        if row["rfm_sum"] >= 10:
            return "top spenders"
        if row["frequency"] == 1 and row["recency"] <= 60:
            return "new customers"
        if row["recency"] > 90 or row["rfm_sum"] <= 5:
            return "at-risk / dormant"
        if row["rfm_sum"] >= 7:
            return "top spenders"
        return "at-risk / dormant"

    grouped["customertype"] = grouped.apply(get_type, axis=1)

    # Discounts
    def discount(ct):
        if ct == "top spenders":
            return "15% vip discount"
        if ct == "new customers":
            return "10% welcome discount"
        if ct == "at-risk / dormant":
            return "25% re-engagement discount"
        return "5% promo"

    grouped["discount"] = grouped["customertype"].apply(discount)

    return grouped


def merge_and_save(df, rfm_table, out_path=output_xl):
    df = df.copy()
    df["customerid"] = df["customerid"].astype(str)
    rfm_table["customerid"] = rfm_table["customerid"].astype(str)

    merged = df.merge(rfm_table, on="customerid", how="left")
    merged.to_excel(out_path, index=False)

    print("Saved:", out_path)
    return merged


def customer_lookup(merged_df):
    cust = input("Enter customer ID: ").strip()

    df = merged_df.copy()
    df["customerid"] = df["customerid"].astype(str).str.strip().str.split(".").str[0]

    cust = cust.strip()
    match = df[df["customerid"] == cust]

    if match.empty:
        print("Customer not found.")
        return

    # Profile section
    profile_cols = [
        "customerid", "recency", "frequency", "monetary",
        "customertype", "discount", "r_score", "f_score",
        "m_score", "rfm_sum"
    ]

    profile = match[profile_cols].drop_duplicates().reset_index(drop=True)

    # Products purchased
    prod_tbl = match.groupby("description").agg({
        "invoicedate": "max",
        "quantity": "sum",
        "totalprice": "sum"
    }).reset_index()

    print("\n===== CUSTOMER PROFILE =====")
    display(profile)

    print("\n===== PRODUCTS PURCHASED =====")
    display(prod_tbl.head(200))

    return profile, prod_tbl


# ============================================================
# 5. RUN PIPELINE
# ============================================================
if __name__ == "__main__":
    df = load_cleaned_csv(input_dir)
    rfm_table = compute_rfm_and_assign(df)
    merged_df = merge_and_save(df, rfm_table, output_xl)
    customer_lookup(merged_df)
