"""
CardioScan — Entraînement du modèle
Script pour préparer les données, entraîner et sauvegarder le modèle Random Forest.

Lancez ce script UNE SEULE FOIS avant de démarrer l'API :
    python train_model.py

Dataset utilisé : Heart Disease UCI (Cleveland)
Téléchargement automatique via ucimlrepo ou depuis Kaggle.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Configuration ──

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
TARGET = 'target'

# ── 1. Chargement des données ───

def load_data():
    """
    Charge le dataset Heart Disease UCI.
    Essaie d'abord via ucimlrepo, sinon depuis une URL publique.
    """
    print("📥 Chargement du dataset Heart Disease UCI...")

    try:
        # Méthode 1 : via la librairie officielle UCI
        from ucimlrepo import fetch_ucirepo
        heart = fetch_ucirepo(id=45)
        X = heart.data.features
        y = heart.data.targets

        df = X.copy()
        df['target'] = (y.values.flatten() > 0).astype(int)  # 0 = sain, 1 = malade
        print(f"✅ Dataset chargé via ucimlrepo — {len(df)} patients")
        return df

    except Exception:
        # Méthode 2 : URL publique Kaggle/UCI
        print("⚠️  ucimlrepo non disponible, chargement depuis URL publique...")
        url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv"
        df = pd.read_csv(url)
        print(f"✅ Dataset chargé depuis URL — {len(df)} patients")
        return df


# ── 2. Préparation des données ─

def prepare_data(df):
    """Nettoyage, encodage et séparation X / y."""
    print("\n🔧 Préparation des données...")

    # Renommer la colonne cible si nécessaire
    if 'target' not in df.columns and 'num' in df.columns:
        df['target'] = (df['num'] > 0).astype(int)

    # Supprimer les lignes avec valeurs manquantes
    df = df.dropna()
    print(f"   → {len(df)} patients après nettoyage")

    # S'assurer que toutes les features sont présentes
    for col in FEATURES:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans le dataset : '{col}'")

    X = df[FEATURES]
    y = df[TARGET]

    print(f"   → Distribution : {y.value_counts().to_dict()}")
    print(f"   → Sains : {(y==0).sum()} | Malades : {(y==1).sum()}")

    return X, y


# ── 3. Entraînement ──

def train_model(X, y):
    """Entraîne un Random Forest et évalue ses performances."""
    print("\n🤖 Entraînement du modèle Random Forest...")

    # Séparation train / test (80% / 20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Modèle Random Forest (meilleur compromis précision / interprétabilité)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)

    # ── Évaluation ──
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n📊 Résultats sur le jeu de test :")
    print(f"   → Accuracy : {acc*100:.1f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Sain', 'Malade'])}")

    # Validation croisée (5 folds)
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    print(f"   → Cross-validation : {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

    # Importance des features
    print(f"\n🔍 Importance des features (Top 5) :")
    importances = pd.Series(model.feature_importances_, index=FEATURES)
    for feat, imp in importances.nlargest(5).items():
        bar = "█" * int(imp * 40)
        print(f"   {feat:<12} {bar} {imp:.3f}")

    return model, scaler


# ── 4. Sauvegarde ───

def save_model(model, scaler):
    """Sauvegarde le modèle et le scaler dans le dossier model/."""
    model_path  = os.path.join(MODEL_DIR, "cardiac_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"\n💾 Modèle sauvegardé → {model_path}")
    print(f"💾 Scaler sauvegardé → {scaler_path}")


# ── 5. Test rapide ──

def quick_test(model, scaler):
    """Teste le modèle sur un exemple réel (patient à risque élevé)."""
    print("\n🧪 Test rapide avec un patient exemple :")

    # Exemple : homme 63 ans, cholestérol 233, angine typique...
    sample = {
        'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
        'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
        'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
    }

    X_sample = np.array([[sample[f] for f in FEATURES]])
    X_scaled  = scaler.transform(X_sample)

    pred  = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]

    print(f"   → Résultat : {'⚠️  RISQUE ÉLEVÉ' if pred==1 else '✅ RISQUE FAIBLE'}")
    print(f"   → Probabilité de maladie : {proba*100:.1f}%")


# ── Main ─

if __name__ == "__main__":
    print("=" * 55)
    print("  CardioScan — Entraînement du modèle")
    print("=" * 55)

    df            = load_data()
    X, y          = prepare_data(df)
    model, scaler = train_model(X, y)
    save_model(model, scaler)
    quick_test(model, scaler)

    print("\n✅ Entraînement terminé ! Lancez maintenant : python app.py")
    print("=" * 55)
