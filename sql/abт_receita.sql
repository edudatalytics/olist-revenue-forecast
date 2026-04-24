SELECT
    strftime('%Y-%m', order_purchase_timestamp) AS ano_mes,
    SUM(price)                                  AS total_vendas,
    SUM(freight_value)                          AS total_frete,
    SUM(price + freight_value)                  AS receita_total,
    COUNT(DISTINCT orders.order_id)             as total_pedidos,
    COUNT(DISTINCT customer_id)                 as total_clientes,
    SUM(price + freight_value) / COUNT(DISTINCT orders.order_id) AS ticket_medio
    FROM orders
    
JOIN items ON orders.order_id = items.order_id
WHERE order_status = 'delivered'
GROUP BY ano_mes
ORDER BY ano_mes