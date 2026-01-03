import os
import numpy as np
import pandas as pd
import xgboost as xgb
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def criar_ia_producao():
    np.random.seed(42)
    n = 2000
    data = {
        'hora': np.random.randint(0, 24, n),
        'dispositivo': np.random.randint(0, 3, n),
        'compras_7d': np.random.randint(0, 5, n),
        'carrinhos': np.random.randint(0, 10, n),
        'valor': np.random.uniform(10, 500, n),
        'comprou': np.random.choice([0, 1], n, p=[0.82, 0.18])
    }
    df = pd.DataFrame(data)
    features = ['hora', 'dispositivo', 'compras_7d', 'carrinhos', 'valor']
    model = xgb.XGBClassifier(n_estimators=20, max_depth=4, learning_rate=0.1)
    model.fit(df[features], df['comprou'])
    return model, features

modelo_ia, colunas_ia = criar_ia_producao()

@app.route("/preve", methods=["POST"])
def preve():
    try:
        dados = request.get_json()
        entrada = pd.DataFrame([{
            'hora': int(dados.get('hora', 12)),
            'dispositivo': int(dados.get('device', 1)),
            'compras_7d': int(dados.get('freq_7d', 0)),
            'carrinhos': int(dados.get('carrinhos', 0)),
            'valor': float(dados.get('valor_ultima', 0))
        }])[colunas_ia]
        prob = float(modelo_ia.predict_proba(entrada)[:, 1][0])
        return jsonify({
            "status": "sucesso",
            "probabilidade": f"{round(prob * 100, 2)}%",
            "recomendacao": "🎯 ALTA CHANCE: FECHAR AGORA!" if prob > 0.55 else "Acompanhar"
        })
    except Exception as e:
        return jsonify({"status": "erro", "msg": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    app.run(host='0.0.0.0', port=port)
