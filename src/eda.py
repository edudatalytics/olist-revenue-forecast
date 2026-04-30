# %%
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import mlflow
import mlflow.sklearn

# %%
df = pd.read_csv("../data/abt_receita.csv")

print(df.shape)
print(df.head())

# %%
#verificando se há valores nulos
df.isnull().sum()

#verificando meses faltantes
todos_os_meses = pd.date_range(start='2016-09', end='2018-08', freq='MS').strftime('%Y-%m')
faltantes = set(todos_os_meses) - set(df['ano_mes'])
print(f'Meses faltantes: {sorted(faltantes)}')

#filtrando meses com menos de 100 pedidos
print(df[df['total_pedidos']<100][['ano_mes', 'total_pedidos', 'receita_total']])

# %% [markdown]
# No mes de 2016-09 e 2016-12 teve apenas um pedido, oque nao é relevante para o modelo, e podendo atrapalhar no aprendizado entao vamos filtrar e usar apenas dados a partir de 2017-01 

# %%
df = df[df['ano_mes']>='2017-01'].reset_index(drop=True)
print(f'Meses restantes:{len(df)}')
print(f'Periodo: {df["ano_mes"].min()} a {df["ano_mes"].max()}')
print(df.head())

# %%
plt.figure(figsize=(12,5))
plt.plot(df['ano_mes'], df['receita_total'],marker='o' , color ='#2ecc71', linewidth=2)
plt.fill_between(range(len(df)), df['receita_total'], alpha = 0.1, color='#2ecc71')
plt.xticks(range(len(df)), df['ano_mes'], rotation=45, ha ='right')
plt.title('Receita Total por Mês', fontweight='bold')
plt.xlabel('Receita Total(R$)')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Observações
# 
# - Tendência clara de crescimento de jan/2017 a ago/2018
# - Pico em novembro/2017 — Black Friday
# - Queda em dezembro/2017 — efeito pós Black Friday, compras antecipadas
# - A partir de 2018 a receita se estabiliza entre R$950k e R$1.1M por mês
# 

# %%
plt.figure(figsize=(12,5))
plt.plot(df['ano_mes'], df['ticket_medio'], marker ='o', color ='#3498db', linewidth=2)
plt.xticks(range(len(df)), df['ano_mes'], rotation=45, ha='right')
plt.title('Ticket Médio por Mês', fontweight='bold')
plt.ylabel('Ticket Médio (R$)')
plt.axhline(y=df['ticket_medio'].mean(), color ='red', linestyle ='--', alpha = 0.7, label =f'Média : R${df["ticket_medio"].mean():.2f}')
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Insight — Ticket Médio vs. Volume
# 
# O ticket médio se manteve estável em torno de R$160 durante todo o período.
# O crescimento da receita foi impulsionado pelo aumento do volume de pedidos,
# não pelo aumento do valor por pedido.
# 
# Isso indica que a estratégia de crescimento da Olist foi baseada em 
# aquisição de novos clientes.
# 

# %%
plt.figure(figsize=(8,6))
corr = df[['receita_total', 'total_pedidos', 'total_clientes', 'ticket_medio']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title('Correlaçao entre Variáveis', fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusão da EDA
# 
# - Correlação de 0.99 entre receita e volume de pedidos
# - Ticket médio tem correlação quase nula com receita (-0.18)
# - O modelo deve focar em prever volume de pedidos como proxy de receita
# 

