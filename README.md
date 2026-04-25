# 📈 Olist Revenue Forecast

Previsão de receita mensal para e-commerce brasileiro usando dados públicos da Olist.  
Pipeline completo: SQL → EDA → Machine Learning → API REST.

---

## 🎯 Resultado

| Modelo | MAE | MAPE |
|---|---|---|
| **Linear Regression** ✅ | **R$ 41.281** | **3.9%** |
| Random Forest | R$ 91.146 | 8.3% |

> MAPE de 3.9% — considerado excelente para previsão de receita mensal.

---

## 🔄 Pipeline

```
Dados brutos (Olist) → SQL (ABT) → EDA → ML → API REST
```

---

## 🗂️ Estrutura

```
olist-revenue-forecast/
├── data/
│   ├── abt_receita.csv         → ABT gerada via SQL
│   └── olist.db                → Banco SQLite local
├── sql/
│   └── abt_receita.sql         → Query de construção da ABT
├── src/
│   ├── eda.py                  → Análise exploratória e visualizações
│   └── treinar.py              → Pipeline de treino e avaliação
├── models/
│   └── modelo_receita.pkl      → Melhor modelo salvo
├── api/
│   └── app.py                  → API REST com FastAPI
├── setup.py                    → Cria o banco SQLite
├── exportar_abt.py             → Exporta ABT para CSV
├── requirements.txt
└── README.md
```

---

## 📊 Principais Insights (EDA)

- Receita cresceu de R$ 127k (jan/2017) para R$ 1.1M (ago/2018)
- **Pico em novembro/2017** — Black Friday
- Ticket médio estável em ~R$ 160 — crescimento impulsionado por volume
- Correlação de 0.99 entre receita e número de pedidos

---

## ▶️ Como Executar

```bash
# 1. Clona o repositório
git clone https://github.com/edudatalytics/olist-revenue-forecast.git
cd olist-revenue-forecast

# 2. Instala dependências
pip install -r requirements.txt

# 3. Baixa o dataset
# https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# Extrai os CSVs na pasta data/

# 4. Cria o banco e exporta a ABT
python setup.py
python exportar_abt.py

# 5. Treina o modelo
python src/treinar.py

# 6. Sobe a API
uvicorn api.app:app --reload
```

---

## 🛠️ Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge)

---

## 👤 Autor

**Eduardo Matos** — Cientista de Dados Júnior  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Eduardo_Matos-blue?style=flat&logo=linkedin)](https://linkedin.com/in/matos-eduardo)
[![GitHub](https://img.shields.io/badge/GitHub-edudatalytics-black?style=flat&logo=github)](https://github.com/edudatalytics)