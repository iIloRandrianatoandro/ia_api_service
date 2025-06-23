from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Charger les modèles au démarrage
model_moulage = joblib.load("./model_inter_donnees/model_moulage.pkl")
scaler_moulage = joblib.load("./model_inter_donnees/scaler_moulage.pkl")

model_soufflage = joblib.load("./model_inter_donnees/model_soufflage.pkl")
scaler_soufflage = joblib.load("./model_inter_donnees/scaler_soufflage.pkl")

model_refroidissement = joblib.load("./model_inter_donnees/model_refroidissement.pkl")
scaler_refroidissement = joblib.load("./model_inter_donnees/scaler_refroidissement.pkl")


def generate_suggestion(machine, etat, temp, pression, vib):
    if etat == 0:  # Normal
        return "Aucune action requise."

    if machine == "moulage":
        if etat == 2 or temp > 250:
            return "Température trop élevée. Vérifiez le système de chauffage."
        if etat >= 1 or pression < 60:
            return "Pression insuffisante. Contrôlez la pompe hydraulique."
        if etat >= 1 or vib > 0.15:
            return "Vibration excessive. Une pièce mécanique peut être usée."

    elif machine == "soufflage":
        if etat == 2 or temp < 90:
            return "Température trop basse. Réglez le four de préchauffage."
        if etat >= 1 or pression > 40:
            return "Pression d’air trop élevée. Vérifiez le compresseur."
        if etat >= 1 or vib > 0.18:
            return "Vibration anormale. Stabilisez la machine."

    elif machine == "refroidissement":
        if etat == 2 or temp > 40:
            return "Température trop élevée. Vérifiez le système de refroidissement."
        if etat >= 1 or pression < 3:
            return "Pression de refroidissement trop basse. Contrôlez la pompe ou le compresseur."
        if etat >= 1 or vib > 0.1:
            return "Vibration anormale. Inspectez les composants mécaniques."

    return "Surveillez la machine de près."


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    machine = data.get("machine")
    temp = data.get("temperature")
    pression = data.get("pression")
    vib = data.get("vibration")

    if None in (machine, temp, pression, vib):
        return jsonify({"error": "Données incomplètes"}), 400

    if machine == "moulage":
        model = model_moulage
        scaler = scaler_moulage
    elif machine == "soufflage":
        model = model_soufflage
        scaler = scaler_soufflage
    elif machine == "refroidissement":
        model = model_refroidissement
        scaler = scaler_refroidissement
    else:
        return jsonify({"error": "Machine inconnue"}), 400

    input_data = np.array([[temp, pression, vib]])
    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)
    etat = int(pred[0])  # 0, 1 ou 2

    etat_labels = {0: "normal", 1: "risque", 2: "critique"}
    result = etat_labels[etat]

    suggestion = generate_suggestion(machine, etat, temp, pression, vib)

    response = {
        "machine": machine,
        "etat": result,
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
