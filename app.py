from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Simule un modèle IA entraîné
model = RandomForestClassifier()
X_train = [[70, 30, 100], [90, 60, 140], [85, 55, 135], [40, 20, 80]]  # température, vibration, pression
y_train = [0, 1, 1, 0]  # 0 = normal, 1 = anomalie
model.fit(X_train, y_train)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = [[data['temperature'], data['vibration'], data['pressure']]]
    prediction = model.predict(features)[0]

    if prediction == 1:
        return jsonify({
            "status": "anomaly",
            "reasons": ["valeurs hors norme"],
            "recommendation": "vérifier le moteur"
        })
    else:
        return jsonify({
            "status": "ok",
            "reasons": [],
            "recommendation": "RAS"
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
