import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# ── Carrega o modelo ──────────────────────────────────────────────────────────
modelo = joblib.load("models/modelo_receita.pkl")

# ── Inicializa a API ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Olist Revenue Forecast API",
    description="Previsão de receita mensal para e-commerce brasileiro",
    version="1.0.0"
)

# ── Schema de entrada ─────────────────────────────────────────────────────────
class DadosEntrada(BaseModel):
    receita_lag1:  float
    pedidos_lag1:  float
    clientes_lag1: float
    ticket_lag1:   float

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {"status": "online", "modelo": "Linear Regression", "mape": "3.9%"}

@app.post("/prever")
def prever(dados: DadosEntrada):
    X = np.array([[
        dados.receita_lag1,
        dados.pedidos_lag1,
        dados.clientes_lag1,
        dados.ticket_lag1
    ]])
    
    previsao = modelo.predict(X)[0]
    
    return {
        "previsao_receita": round(previsao, 2),
        "moeda": "BRL",
        "modelo": "Linear Regression",
        "mape": "3.9%"
    }