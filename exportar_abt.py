#%%
import pandas as pd
import sqlite3

# Conecta ao banco de dados
conn = sqlite3.connect('data/olist.db')

#fazebdo a leitura da query SQL
query = """

SELECT
    strftime('%Y-%m', order_purchase_timestamp)              AS ano_mes,
    SUM(price)                                               AS total_vendas,
    SUM(freight_value)                                       AS total_frete,
    SUM(price + freight_value)                               AS receita_total,
    COUNT(DISTINCT orders.order_id)                          AS total_pedidos,
    COUNT(DISTINCT customer_id)                              AS total_clientes,
    SUM(price + freight_value) / COUNT(DISTINCT orders.order_id) AS ticket_medio
FROM orders
JOIN items ON orders.order_id = items.order_id
WHERE order_status = 'delivered'
GROUP BY ano_mes
ORDER BY ano_mes
"""

# Executa a query e armazena o resultado em um DataFrame
abt = pd.read_sql_query(query, conn)
abt.to_csv('data/abt_receita.csv',index=False)
print("ABT exportada: data/abt_receita.csv")
print(abt.head(10))


