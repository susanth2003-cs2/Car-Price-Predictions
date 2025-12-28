from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# ================= LOAD MODEL & SCALER =================
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    FEATURES = pickle.load(f)

# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Map input in correct feature order, fill missing ones with 0
        input_data = [data.get(col, 0) for col in FEATURES]

        input_array = np.array(input_data).reshape(1, -1)

        # Scale numeric features (assumes scaler was trained on numeric cols in FEATURES)
        input_scaled = scaler.transform(input_array)

        # Predict
        prediction = model.predict(input_scaled)[0]

        return jsonify({
            "price": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

# ================= MAIN =================
if __name__ == "__main__":
    app.run(debug=True)
