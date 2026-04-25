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

# %% [markdown]
# vamos iniciar a modelagem!
# 
# como queremos pever valores vamos usar o modelo de regressao 
# 
# vamos usar a tecnica lag features, usando os valores do mes anterior para prever o mes atual 
# 

# %%
# Criando lag features — valores do mês anterior para cada variável
df['receita_lag1'] = df['receita_total'].shift(1)
df['pedidos_lag1'] = df['total_pedidos'].shift(1)
df['clientes_lag1'] = df['total_clientes'].shift(1)
df['ticket_lag1'] = df['ticket_medio'].shift(1)

# Removendo a primeira linha que agora tem valores nulos devido ao shift
df =df.dropna().reset_index(drop=True)

print(f'shepe após criação de lags: {df.shape}')
print(df[['ano_mes', 'receita_total', 'receita_lag1', 'total_pedidos', 'pedidos_lag1', 'total_clientes', 'clientes_lag1', 'ticket_medio', 'ticket_lag1']].head())

# %% [markdown]
# O shift(1) "empurrou" os valores um mês para frente. Agora o modelo pode usar o que aconteceu em março para prever abril.
# 

# %%
X = df[['receita_lag1', 'pedidos_lag1', 'clientes_lag1', 'ticket_lag1']]

y = df['receita_total']

print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(X.head())


# %% [markdown]
# Vamos usar os primeiros 14 meses para treino e os últimos 3 para teste
# 

# %%
X_train = X.iloc[:14]
X_test = X.iloc[14:]

y_train = y.iloc[:14]
y_test = y.iloc[14:]

print(f'Treino: {len(X_train)} meses')
print(f'Teste: {len(X_test)} meses')
print(f'\nMeses de teste: {df["ano_mes"].iloc[14:].values}')


# %% [markdown]
# vamos criar um modelo de regressão simples usando Random Forest para prever a receita total com base nas features de lag.
# 

# %%
modelo = RandomForestRegressor(n_estimators = 100 , random_state = 42)
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)

mape = mean_absolute_percentage_error(y_test, y_pred) *100

print(f'MAE: R${mae:.2f}')
print(f"MAPE: {mape:.1f}%")

# Compara real vs previsto
resultado = pd.DataFrame({
    "mes":      df["ano_mes"].iloc[14:].values,
    "real":     y_test.values,
    "previsto": y_pred
})
print(resultado)


# %% [markdown]
# ## Limitações do Modelo
# 
# - O modelo superestima a receita de 2018 em ~11.8%
# - Motivo: foi treinado com dados de 2017 onde a tendência era de crescimento constante
# - A receita se estabilizou em 2018, comportamento que o modelo não capturou bem
# - Com mais dados (2019, 2020) o modelo aprenderia melhor a estabilização

# %%
plt.figure(figsize=(10, 5))
plt.plot(resultado["mes"], resultado["real"],     
         marker="o", label="Real",     color="#27ae60", linewidth=2)
plt.plot(resultado["mes"], resultado["previsto"], 
         marker="o", label="Previsto", color="#e74c3c", linewidth=2, linestyle="--")
plt.title("Real vs Previsto — Receita Mensal", fontweight="bold")
plt.ylabel("Receita (R$)")
plt.legend()
plt.tight_layout()
plt.show()

# %%


# Inicia o experimento
mlflow.set_experiment("olist-revenue-forecast")

with mlflow.start_run(run_name="random-forest"):
    
    # Loga os parâmetros
    mlflow.log_param("modelo", "RandomForestRegressor")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("features", list(X.columns))
    mlflow.log_param("meses_treino", len(X_train))
    mlflow.log_param("meses_teste", len(X_test))
    
    # Loga as métricas
    mlflow.log_metric("mae",  mae)
    mlflow.log_metric("mape", mape)
    
    # Loga o modelo
    mlflow.sklearn.log_model(modelo, "random-forest-model")
    
    print("Experimento registrado no MLflow!")
    print(f"MAE:  R$ {mae:,.2f}")
    print(f"MAPE: {mape:.1f}%")


