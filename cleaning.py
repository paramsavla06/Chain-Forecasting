import os
import pandas as pd


def clean_record(df):
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
        'country': 'country'
    }
    df = df.rename(columns=rename)

    df = df[~df['invoiceno'].astype(str).str.startswith("C")]
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df = df.dropna(subset=['invoicedate'])
    df = df.dropna(subset=['customerid'])
    df['customerid'] = df['customerid'].astype(str)
    df = df[(df['quantity'] > 0) & (df['price'] > 0)]
    df['description'] = df['description'].astype(str).str.strip()
    df = df[(df['description'] != "") & (~df['description'].str.isnumeric())]
    df['sales'] = df['quantity'] * df['price']

    return df.reset_index(drop=True)


if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "dataset")
    raw_path = os.path.join(dataset_path, "online_retail.csv")
    clean_path = os.path.join(dataset_path, "cleaned_online_retail.xlsx")

    print("Searching for raw CSV file at:", raw_path)

    if not os.path.exists(raw_path):
        print("\nERROR: online_retail.csv not found in: ", base_dir)
        print("\nPlace the raw CSV in the SAME folder as cleaning.py and run again.")
        exit()

    print("\nLoading raw CSV...")
    df = pd.read_csv(raw_path, encoding="latin1", low_memory=False)

    print("Cleaning data...")
    df = clean_record(df)

    print("Saving cleaned dataset to:", clean_path)
    df.to_excel(clean_path, index=False)

    print("Done! Saved the clean datasheet at: ", clean_path)
