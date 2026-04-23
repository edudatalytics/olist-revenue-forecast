import sqlite3
import pandas as pd

# Carrega os CSVs
orders = pd.read_csv("data/olist_orders_dataset.csv")
items  = pd.read_csv("data/olist_order_items_dataset.csv")

# Cria o banco .db
conn = sqlite3.connect("data/olist.db")
orders.to_sql("orders", conn, index=False, if_exists="replace")
items.to_sql("items",   conn, index=False, if_exists="replace")
conn.close()

print("Banco criado: data/olist.db")
print(f"Tabelas: orders ({len(orders):,} linhas), items ({len(items):,} linhas)")