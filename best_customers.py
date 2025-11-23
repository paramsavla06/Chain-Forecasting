import os
import pandas as pd
import numpy as np
from datetime import timedelta
from IPython.display import display

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, "dataset")

clean_xl = os.path.join(dataset_dir, "cleaned_online_retail.xlsx")
rfm_xl = os.path.join(dataset_dir, "cleaned_online_retail_rfm.xlsx")
output_xl = os.path.join(dataset_dir, "top_customers.xlsx")

TOP_N = 50
MIN_TX_FOR_TOP = 1


def load_dataset(prefer_rfm=True):
    if prefer_rfm and os.path.exists(rfm_xl):
        print("Loading existing RFM-enriched file:", rfm_xl)
        df = pd.read_excel(rfm_xl, engine="openpyxl")
    else:
        if not os.path.exists(clean_xl):
            raise FileNotFoundError(f"Cleaned Excel not found at {clean_xl}")
        print("Loading cleaned file:", clean_xl)
        df = pd.read_excel(clean_xl, engine="openpyxl")

    # convert all columns to lowercase
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    return df


def compute_rfm_if_needed(df):

    # Ensure required invoice date exists
    if "invoicedate" not in df.columns:
        raise ValueError("Missing 'invoicedate' column.")
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")

    # Normalize customerid
    if "customerid" not in df.columns:
        df["customerid"] = np.nan

    df["customerid"] = (
        df["customerid"]
        .astype(str)
        .str.strip()
        .str.split(".").str[0]  # remove decimals like 17850.0
    )

    # Make sure Quantity and Price are numeric
    if "quantity" in df.columns:
        df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    else:
        df["quantity"] = np.nan

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = np.nan

    # Fix TotalPrice / Monetary column
    if "totalprice" in df.columns:
        pass
    elif "sales" in df.columns:
        df.rename(columns={"sales": "totalprice"}, inplace=True)
    else:
        df["totalprice"] = df["quantity"] * df["price"]

    # KEEP ONLY valid rows
    df = df[df["invoicedate"].notna() & df["totalprice"].notna()].copy()

    # If RFM columns already exist → reuse them
    required = {
        "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score",
        "rfm_sum", "customertype", "discount"
    }

    if required.issubset(df.columns):
        print("Existing RFM detected — using file RFM.")
        rfm_table = df[["customerid", *required]].drop_duplicates("customerid")
        return df, rfm_table

    # Otherwise compute RFM fresh
    print("Computing RFM…")

    df_has = df[df["customerid"].notna()].copy()
    snapshot = df_has["invoicedate"].max() + timedelta(days=1)

    # detect invoice number column
    invoice_col = None
    for c in ["invoiceno", "invoice", "invoicenumber", "invoiceno."]:
        if c in df_has.columns:
            invoice_col = c
            break

    if invoice_col:
        agg = df_has.groupby("customerid").agg({
            "invoicedate": lambda x: (snapshot - x.max()).days,
            invoice_col: "nunique",
            "totalprice": "sum"
        }).rename(columns={
            "invoicedate": "recency",
            invoice_col: "frequency",
            "totalprice": "monetary"
        })
    else:
        agg = df_has.groupby("customerid").agg({
            "invoicedate": lambda x: (snapshot - x.max()).days,
            "totalprice": "sum"
        }).rename(columns={
            "invoicedate": "recency",
            "totalprice": "monetary"
        })
        agg["frequency"] = df_has.groupby("customerid").size()

    agg = agg.reset_index()

    # RFM SCORES
    agg["r_score"] = pd.qcut(
        agg["recency"].rank(method="first"), 4,
        labels=[4, 3, 2, 1]
    ).astype(int)

    agg["f_score"] = pd.qcut(
        agg["frequency"].rank(method="first"), 4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    agg["m_score"] = pd.qcut(
        agg["monetary"].rank(method="first"), 4,
        labels=[1, 2, 3, 4]
    ).astype(int)

    agg["rfm_sum"] = agg["r_score"] + agg["f_score"] + agg["m_score"]

    # CUSTOMER TYPE LABELS
    def assign_type(row):
        if row["rfm_sum"] >= 10:
            return "top spenders"
        if row["frequency"] == 1 and row["recency"] <= 60:
            return "new customers"
        if row["recency"] > 90 or row["rfm_sum"] <= 5:
            return "at-risk / dormant"
        if row["rfm_sum"] >= 7:
            return "top spenders"
        return "at-risk / dormant"

    agg["customertype"] = agg.apply(assign_type, axis=1)

    # DISCOUNTS
    agg["discount"] = agg["customertype"].map({
        "top spenders": "15% vip discount",
        "new customers": "10% welcome discount",
        "at-risk / dormant": "25% re-engagement discount"
    })

    return df, agg


def top_customers_table(rfm_table, top_n=TOP_N, min_tx=MIN_TX_FOR_TOP):

    if "frequency" in rfm_table.columns:
        rfm_f = rfm_table[rfm_table["frequency"] >= min_tx].copy()
    else:
        rfm_f = rfm_table.copy()

    sort_cols = []
    if "rfm_sum" in rfm_f.columns:
        sort_cols.append(("rfm_sum", False))
    if "monetary" in rfm_f.columns:
        sort_cols.append(("monetary", False))
    if "frequency" in rfm_f.columns:
        sort_cols.append(("frequency", False))
    if "recency" in rfm_f.columns:
        sort_cols.append(("recency", True))

    sort_by = [c for c, _ in sort_cols]
    ascending = [a for _, a in sort_cols]

    rfm_sorted = rfm_f.sort_values(sort_by, ascending=ascending)
    rfm_sorted.index = range(1, len(rfm_sorted) + 1)

    keep = [
        "customerid", "recency", "frequency", "monetary",
        "r_score", "f_score", "m_score",
        "rfm_sum", "customertype", "discount"
    ]
    keep = [c for c in keep if c in rfm_sorted.columns]

    return rfm_sorted[keep].head(top_n)


if __name__ == "__main__":

    df = load_dataset(prefer_rfm=True)
    df, rfm_table = compute_rfm_if_needed(df)
    top_tbl = top_customers_table(rfm_table)

    top_tbl.to_excel(output_xl, index=False)
    print(f"\nSaved Top-{TOP_N} Customers → {output_xl}")

    print("\n--- ABOUT THE RANKING ---")
    print("• Lower recency (recent purchases) → better")
    print("• Higher frequency → better")
    print("• Higher monetary → better")
    print("• Ranked by: RFM_Sum → Monetary → Frequency → Recency\n")

    display(top_tbl)
