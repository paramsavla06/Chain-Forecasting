import os
import pandas as pd
from utils import forecast_sales
from cleaning import clean_record

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_dir = os.path.join(base_dir, "dataset")
path = os.path.join(dataset_dir, "online_retail.csv")

df = pd.read_csv(path, encoding="latin1", low_memory=False)
df = clean_record(df)

product_input = input()
forecast_sales(df, product_input)
