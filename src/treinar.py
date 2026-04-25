import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

import joblib
import os

# ── 1. Carrega os dados ──────────────────────────────────────────────────────
df = pd.read_csv("data/abt_receita.csv")

# ── 2. Filtra período válido ─────────────────────────────────────────────────
df = df[df["ano_mes"] >= "2017-01"].reset_index(drop=True)

# ── 3. Cria lag features ─────────────────────────────────────────────────────
df["receita_lag1"]  = df["receita_total"].shift(1)
df["pedidos_lag1"]  = df["total_pedidos"].shift(1)
df["clientes_lag1"] = df["total_clientes"].shift(1)
df["ticket_lag1"]   = df["ticket_medio"].shift(1)
df = df.dropna().reset_index(drop=True)

# ── 4. Define X e y ──────────────────────────────────────────────────────────
FEATURES = ["receita_lag1", "pedidos_lag1", "clientes_lag1", "ticket_lag1"]
TARGET   = "receita_total"

X = df[FEATURES]
y = df[TARGET]

# ── 5. Split temporal ────────────────────────────────────────────────────────
X_train, X_test = X.iloc[:-5], X.iloc[-5:]
y_train, y_test = y.iloc[:-5], y.iloc[-5:]

print(f"Treino: {len(X_train)} meses | Teste: {len(X_test)} meses")

modelos = {
    "RandomForest": RandomForestRegressor(n_estimators =100 , random_state=420),
    "LinearRegression": LinearRegression()
}

resultados = []

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    resultados.append({
        "modelo": nome,
        "MAE": mae,
        "MAPE": mape
    })

    print(f"{nome:20} → MAE: R$ {mae:,.2f} | MAPE: {mape:.1f}%")

# ── 8. Melhor modelo ─────────────────────────────────────────────────────────
df_resultados = pd.DataFrame(resultados)
melhor = df_resultados.loc[df_resultados["MAPE"].idxmin(), "modelo"]

print(f"\nMelhor modelo: {melhor}")

os.makedirs('models', exist_ok=True)
melhor_modelo = modelos[melhor]
joblib.dump(melhor_modelo, "models/modelo_receita.pkl")
print(f"Modelo salvo em models/modelo_receita.pkl")

