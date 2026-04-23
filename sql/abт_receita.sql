select *
FROM ORDERS
JOIN items on orders.order_id = items.order_id
WHERE order_status = 'delivered'