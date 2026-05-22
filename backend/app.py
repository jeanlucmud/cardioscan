"""
CardioScan — Backend Flask
API REST pour la prédiction du risque cardiaque
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Autorise les appels depuis le frontend HTML

# Charger le modèle entraîné
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cardiac_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Ordre exact des features (doit correspondre à l'entraînement)
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint principal de prédiction.
    Reçoit un JSON avec les 13 paramètres cliniques.
    Retourne le risque (HIGH/LOW), la probabilité et les facteurs.
    """
    try:
        data = request.get_json()

        # Validation des champs
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Champs manquants : {missing}"}), 400

        # Construction du vecteur de features
        X = np.array([[data[f] for f in FEATURES]])
        X_scaled = scaler.transform(X)

        # Prédiction
        prediction = model.predict(X_scaled)[0]           # 0 = sain, 1 = malade
        proba = model.predict_proba(X_scaled)[0][1]       # Proba d'être malade

        risk = "HIGH" if prediction == 1 else "LOW"
        probability = round(float(proba) * 100, 1)

        # Génération des facteurs d'analyse
        factors = analyze_factors(data, risk)

        return jsonify({
            "risk": risk,
            "probability": probability,
            "factors": factors
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def analyze_factors(data, risk):
    """Génère une liste de facteurs de risque lisibles pour l'utilisateur."""
    factors = []

    # Âge
    age = data['age']
    if age >= 60:
        factors.append({"status": "warn", "text": f"Âge élevé ({age} ans) — facteur de risque majeur"})
    elif age >= 45:
        factors.append({"status": "info", "text": f"Âge intermédiaire ({age} ans) — surveillance conseillée"})
    else:
        factors.append({"status": "ok", "text": f"Âge favorable ({age} ans)"})

    # Cholestérol
    chol = data['chol']
    if chol > 240:
        factors.append({"status": "warn", "text": f"Cholestérol élevé ({chol} mg/dl) — risque accru"})
    elif chol > 200:
        factors.append({"status": "info", "text": f"Cholestérol limite ({chol} mg/dl) — à surveiller"})
    else:
        factors.append({"status": "ok", "text": f"Cholestérol normal ({chol} mg/dl)"})

    # Pression artérielle
    bp = data['trestbps']
    if bp > 140:
        factors.append({"status": "warn", "text": f"Hypertension ({bp} mmHg) — facteur cardiovasculaire"})
    elif bp > 120:
        factors.append({"status": "info", "text": f"Pression légèrement élevée ({bp} mmHg)"})
    else:
        factors.append({"status": "ok", "text": f"Pression artérielle normale ({bp} mmHg)"})

    # Fréquence cardiaque max
    thalach = data['thalach']
    if thalach < 100:
        factors.append({"status": "warn", "text": f"Fréquence cardiaque max faible ({thalach} bpm)"})
    elif thalach > 150:
        factors.append({"status": "ok", "text": f"Bonne capacité cardiaque ({thalach} bpm)"})

    # Angine à l'effort
    if data['exang'] == 1:
        factors.append({"status": "warn", "text": "Angine induite à l'effort — signe d'ischémie"})
    else:
        factors.append({"status": "ok", "text": "Pas d'angine à l'effort"})

    # Dépression ST
    oldpeak = data['oldpeak']
    if oldpeak > 2:
        factors.append({"status": "warn", "text": f"Dépression ST significative ({oldpeak}) — ischémie probable"})
    elif oldpeak > 0:
        factors.append({"status": "info", "text": f"Légère dépression ST ({oldpeak})"})

    # Vaisseaux colorés
    ca = data['ca']
    if ca >= 2:
        factors.append({"status": "warn", "text": f"{ca} vaisseau(x) obstrué(s) à la fluoroscopie"})
    elif ca == 1:
        factors.append({"status": "info", "text": "1 vaisseau avec anomalie détectée"})
    else:
        factors.append({"status": "ok", "text": "Aucun vaisseau obstrué détecté"})

    return factors[:6]  # Limiter à 6 facteurs


@app.route("/health", methods=["GET"])
def health():
    """Vérification de l'état du serveur."""
    return jsonify({"status": "ok", "model": "Random Forest", "version": "1.0"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
