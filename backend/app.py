"""
CardioScan — Backend Flask
API REST pour la prédiction du risque cardiaque
Le modèle s'entraîne automatiquement s'il n'existe pas.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_DIR   = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "cardiac_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

model  = None
scaler = None


def train_and_save():
    """Entraîne le modèle et le sauvegarde si absent."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    print("🤖 Modèle absent — entraînement en cours...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Téléchargement du dataset
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
    df  = pd.read_csv(url)
    print(f"✅ Dataset chargé — {len(df)} patients")

    X = df[FEATURES]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)

    clf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_train_sc, y_train)

    joblib.dump(clf, MODEL_PATH)
    joblib.dump(sc,  SCALER_PATH)
    print("💾 Modèle sauvegardé !")
    return clf, sc


def load_model():
    """Charge ou entraîne le modèle au démarrage."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print("✅ Chargement du modèle existant...")
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        model, scaler = train_and_save()
    print("🚀 API prête !")


# Charger le modèle au démarrage
load_model()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Champs manquants : {missing}"}), 400

        X        = np.array([[data[f] for f in FEATURES]])
        X_scaled = scaler.transform(X)

        prediction = model.predict(X_scaled)[0]
        proba      = model.predict_proba(X_scaled)[0][1]

        risk        = "HIGH" if prediction == 1 else "LOW"
        probability = round(float(proba) * 100, 1)
        factors     = analyze_factors(data, risk)

        return jsonify({"risk": risk, "probability": probability, "factors": factors})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def analyze_factors(data, risk):
    factors = []

    age = data['age']
    if age >= 60:
        factors.append({"status": "warn", "text": f"Âge élevé ({age} ans) — facteur de risque majeur"})
    elif age >= 45:
        factors.append({"status": "info", "text": f"Âge intermédiaire ({age} ans) — surveillance conseillée"})
    else:
        factors.append({"status": "ok", "text": f"Âge favorable ({age} ans)"})

    chol = data['chol']
    if chol > 240:
        factors.append({"status": "warn", "text": f"Cholestérol élevé ({chol} mg/dl) — risque accru"})
    elif chol > 200:
        factors.append({"status": "info", "text": f"Cholestérol limite ({chol} mg/dl) — à surveiller"})
    else:
        factors.append({"status": "ok", "text": f"Cholestérol normal ({chol} mg/dl)"})

    bp = data['trestbps']
    if bp > 140:
        factors.append({"status": "warn", "text": f"Hypertension ({bp} mmHg) — facteur cardiovasculaire"})
    elif bp > 120:
        factors.append({"status": "info", "text": f"Pression légèrement élevée ({bp} mmHg)"})
    else:
        factors.append({"status": "ok", "text": f"Pression artérielle normale ({bp} mmHg)"})

    thalach = data['thalach']
    if thalach < 100:
        factors.append({"status": "warn", "text": f"Fréquence cardiaque max faible ({thalach} bpm)"})
    elif thalach > 150:
        factors.append({"status": "ok", "text": f"Bonne capacité cardiaque ({thalach} bpm)"})

    if data['exang'] == 1:
        factors.append({"status": "warn", "text": "Angine induite à l'effort — signe d'ischémie"})
    else:
        factors.append({"status": "ok", "text": "Pas d'angine à l'effort"})

    oldpeak = data['oldpeak']
    if oldpeak > 2:
        factors.append({"status": "warn", "text": f"Dépression ST significative ({oldpeak}) — ischémie probable"})
    elif oldpeak > 0:
        factors.append({"status": "info", "text": f"Légère dépression ST ({oldpeak})"})

    ca = data['ca']
    if ca >= 2:
        factors.append({"status": "warn", "text": f"{ca} vaisseau(x) obstrué(s) à la fluoroscopie"})
    elif ca == 1:
        factors.append({"status": "info", "text": "1 vaisseau avec anomalie détectée"})
    else:
        factors.append({"status": "ok", "text": "Aucun vaisseau obstrué détecté"})

    return factors[:6]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "Random Forest", "version": "1.0"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
