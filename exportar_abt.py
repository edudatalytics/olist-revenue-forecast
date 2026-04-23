#%%
import pandas as pd
import sqlite3

# Conecta ao banco de dados
conn = sqlite3.connect('data/olist.db')

#fazebdo a leitura da query SQL
query = """

select
    strftime('%Y-%m', order_purchase_timestamp) as ano_mes,
    SUM(price) as total_vendas,
    SUM(freight_value) as total_frete,
    SUM(price + freight_value) as receita_total
from orders

join items ON orders.order_id = items.order_id
WHERE order_status = 'delivered'
GROUP BY ano_mes
ORDER BY ano_mes

"""

# Executa a query e armazena o resultado em um DataFrame
abt = pd.read_sql_query(query, conn)
abt.to_csv('data/abt_receita.csv',index=False)
print("ABT exportada: data/abt_receita.csv")
print(abt.head(10))


