# fichier : app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger les modèles au démarrage
model_moulage = joblib.load("./model_inter_donnees/model_moulage.pkl")
scaler_moulage = joblib.load("./model_inter_donnees/scaler_moulage.pkl")
model_soufflage = joblib.load("./model_inter_donnees/model_soufflage.pkl")
scaler_soufflage = joblib.load("./model_inter_donnees/scaler_soufflage.pkl")

def generate_suggestion(machine, temp, pression, vib):
    if machine == "moulage":
        if temp > 250:
            return "Température trop élevée. Vérifiez le système de chauffage."
        elif pression < 60:
            return "Pression insuffisante. Contrôlez la pompe hydraulique."
        elif vib > 0.15:
            return "Vibration excessive. Une pièce mécanique peut être usée."
        else:
            return "Aucune action requise."
    elif machine == "soufflage":
        if temp < 90:
            return "Température trop basse. Réglez le four de préchauffage."
        elif pression > 40:
            return "Pression d’air trop élevée. Vérifiez le compresseur."
        elif vib > 0.18:
            return "Vibration anormale. Stabilisez la machine."
        else:
            return "Aucune action requise."

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    machine = data.get("machine")
    temp = data.get("temperature")
    pression = data.get("pression")
    vib = data.get("vibration")

    if machine == "moulage":
        model = model_moulage
        scaler = scaler_moulage
    elif machine == "soufflage":
        model = model_soufflage
        scaler = scaler_soufflage
    else:
        return jsonify({"error": "Machine inconnue"}), 400

    input_data = np.array([[temp, pression, vib]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    is_anomaly = (pred[0] == -1)

    if is_anomaly:
        result = "Anomalie détectée"
        suggestion = generate_suggestion(machine, temp, pression, vib)
    else:
        result = "Aucune anomalie"
        suggestion = "Aucune action requise."

    response = {
        "machine": machine,
        "resultat": result,
        "suggestion": suggestion,
        "donnees": {
            "temperature": temp,
            "pression": pression,
            "vibration": vib
        }
    }
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
