import os
import pandas as pd
from IPython.display import display

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "dataset")
input_dir = os.path.join(data_dir, "cleaned_online_retail.xlsx")

# Load Excel
df = pd.read_excel(input_dir, engine="openpyxl")

df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]

# Fix customerid (remove .0 issue)
if "customerid" in df.columns:
    df["customerid"] = (
        df["customerid"]
        .astype(str)
        .str.strip()
        .str.split(".").str[0]
    )
else:
    df["customerid"] = ""

# Ensure description exists
if "description" not in df.columns:
    df["description"] = ""

df["description"] = df["description"].astype(str).str.strip()

df["quantity"] = pd.to_numeric(
    df.get("quantity", pd.Series()), errors="coerce")
df["price"] = pd.to_numeric(df.get("price", pd.Series()), errors="coerce")

if "totalprice" not in df.columns:
    df["totalprice"] = df["quantity"] * df["price"]

df = df[df["totalprice"].notna() & (df["totalprice"] > 0)]

product_stats = (
    df.groupby("description")
      .agg(
          total_sales=("totalprice", "sum"),
          total_quantity=("quantity", "sum"),
          unique_customers=("customerid", "nunique")
    )
    .reset_index()
)

top_products = product_stats.sort_values("total_sales", ascending=False)

print("\n====== TOP 20 BEST-SELLING PRODUCTS ======")
display(top_products.head(20))

worst_products = product_stats.sort_values("total_sales", ascending=True)

print("\n====== BOTTOM 20 WORST-SELLING PRODUCTS ======")
display(worst_products.head(20))
